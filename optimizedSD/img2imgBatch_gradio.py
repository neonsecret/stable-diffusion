import argparse
import logging
import mimetypes
import os
import re
import sys
import time
from contextlib import nullcontext
from itertools import islice
from random import randint

import gradio as gr
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging as transformers_logging

from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts
import asyncio

transformers_logging.set_verbosity_error()

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

promptRegex = "@q@"


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(image, h0, w0):
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    print("resize done, loading image to torch..")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


async def get_logs():
    return "\n".join([x for x in open("log.txt", "r", encoding="utf8").readlines()] +
                     [y for y in open("tqdm.txt", "r", encoding="utf8").readlines()])


async def get_nvidia_smi():
    proc = await asyncio.create_subprocess_shell('nvidia-smi', stdout=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return str(stdout)


def generate(
        image,
        prompt,
        strength,
        ddim_steps,
        n_iter,
        batch_size,
        Height,
        Width,
        scale,
        ddim_eta,
        unet_bs,
        device,
        seed,
        outdir,
        img_format,
        turbo,
        full_precision,
        sampler,
        speed_mp
):

    oSeed = seed
    oPrompt = prompt


    subfolders = [f.path for f in os.scandir(image) if f.is_dir()]


    for input_dir in subfolders:


        promptword = input_dir.rsplit("\\", 1)[-1] or input_dir
        prompt = oPrompt.replace(promptRegex, promptword)


        print(prompt)

        logging.info(f"prompt: {prompt}, W: {Width}, H: {Height}")
        src_images = [file for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)] if
                      os.path.isfile(file)]

        for image in src_images:
            image = Image.open(image)
            init_image = load_img(image, Height, Width).to(device)
            model.unet_bs = unet_bs
            model.turbo = turbo
            model.cdevice = device
            modelCS.cond_stage_model.device = device


            #seed = oSeed

            try:
                seed = int(oSeed)
            except:
                seed = randint(0, 1000000)


            if device != "cpu" and not full_precision:
                model.half()
                modelCS.half()
                modelFS.half()
                init_image = init_image.half()

            tic = time.time()
            os.makedirs(outdir, exist_ok=True)
            outpath = outdir
            sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt)))[:150]
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))

            # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
            assert prompt is not None
            data = [batch_size * [prompt]]

            modelFS.to(device)

            init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
            init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

            if device != "cpu":
                mem = torch.cuda.memory_allocated() / 1e6
                modelFS.to("cpu")
                while torch.cuda.memory_allocated() / 1e6 >= mem:
                    time.sleep(1)

            assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
            t_enc = int(strength * ddim_steps)
            print(f"target t_enc is {t_enc} steps")

            if not full_precision and device != "cpu":
                precision_scope = autocast
            else:
                precision_scope = nullcontext

            all_samples = []
            seeds = ""
            with torch.no_grad():
                for _ in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        with precision_scope("cuda"):
                            modelCS.to(device)
                            uc = None
                            if scale != 1.0:
                                uc = modelCS.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)

                            subprompts, weights = split_weighted_subprompts(prompts[0])
                            if len(subprompts) > 1:
                                c = torch.zeros_like(uc)
                                totalWeight = sum(weights)
                                # normalize each "sub prompt" and add it
                                for i in range(len(subprompts)):
                                    weight = weights[i]
                                    # if not skip_normalize:
                                    weight = weight / totalWeight
                                    c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                            else:
                                c = modelCS.get_learned_conditioning(prompts)

                            if device != "cpu":
                                mem = torch.cuda.memory_allocated() / 1e6
                                modelCS.to("cpu")
                                while torch.cuda.memory_allocated() / 1e6 >= mem:
                                    time.sleep(1)

                            # encode (scaled latent)
                            z_enc = model.stochastic_encode(
                                init_latent, torch.tensor([t_enc] * batch_size).to(device), seed, ddim_eta, ddim_steps
                            )
                            # decode it
                            samples_ddim = model.sample(
                                t_enc,
                                c,
                                z_enc,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                                sampler=sampler,
                                speed_mp=speed_mp,
                                batch_size=batch_size
                            )

                            modelFS.to(device)
                            print("saving images")
                            for i in range(batch_size):
                                x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                all_samples.append(x_sample.to("cpu"))
                                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{img_format}")
                                )
                                seeds += str(seed) + ","
                                seed += 1
                                base_count += 1

                            if device != "cpu":
                                mem = torch.cuda.memory_allocated() / 1e6
                                modelFS.to("cpu")
                                while torch.cuda.memory_allocated() / 1e6 >= mem:
                                    time.sleep(1)

                            del samples_ddim
                            del x_sample
                            del x_samples_ddim
                            print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

            toc = time.time()

            time_taken = (toc - tic) / 60.0
            grid = torch.cat(all_samples, 0)
            grid = make_grid(grid, nrow=n_iter)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

            txt = (
                    "Samples finished in "
                    + str(round(time_taken, 3))
                    + " minutes and exported to \n"
                    + sample_path
                    + "\nSeeds used = "
                    + seeds[:-1]
            )
    return Image.fromarray(grid.astype(np.uint8)), txt


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


if __name__ == '__main__':
    global lines
    lines = []
    file_handler = logging.FileHandler(filename='log.txt', mode='w')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler, TqdmLoggingHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    parser = argparse.ArgumentParser(description='img2img using gradio')
    parser.add_argument('--config_path', default="optimizedSD/v1-inference.yaml", type=str, help='config path')
    parser.add_argument('--ckpt_path', default="models/ldm/stable-diffusion-v1/model.ckpt", type=str, help='ckpt path')
    parser.add_argument('--outputs_path', default="outputs/img2img-samples", type=str, help='output imgs path')
    args = parser.parse_args()
    config = args.config_path
    ckpt = args.ckpt_path
    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    demo = gr.Blocks()

    with demo:
        with gr.Column():
            gr.Markdown("# img2img batch processing (qdiiibp's adjustments)")
            gr.Markdown("### !! enter the path to the parent folder containing subfolders containing images to process")
            gr.Markdown("### !! \" "+ promptRegex + " \" in the prompt will be replaced by the names of the subfolders containing images to process")
            gr.Markdown("### Press 'print logs' button to get the model output logs")
            with gr.Row():
                with gr.Column():
                    outs1 = [gr.Image(label="Output Image"), gr.Text(label="Generation results")]
                    outs2 = [gr.Text(label="Logs")]
                    outs3 = [gr.Text(label="nvidia-smi")]
                    b1 = gr.Button("Generate!")
                    b2 = gr.Button("Print logs")
                    b3 = gr.Button("nvidia-smi")
                with gr.Column():
                    with gr.Box():
                        b1.click(generate, inputs=[
                            ##gr.Image(tool="editor", type="pil", label="Initial image"),
                            gr.Text(label="path to parent folder containing subfolders containing the images to process"),
                            gr.Text(label="Your Prompt, \""+ promptRegex + " \" will be replaced by folder names", value="an image of a "+ promptRegex + " with a hat"),
                            gr.Slider(0, 1, value=0.85, label="Generated image strength"),
                            gr.Slider(1, 1000, value=20, label="Sampling Steps"),
                            gr.Slider(1, 100, step=1, label="Number of images"),
                            gr.Slider(1, 100, step=1, label="Batch size"),
                            gr.Slider(64, 4096, value=512, step=64, label="Height"),
                            gr.Slider(64, 4096, value=512, step=64, label="Width"),
                            gr.Slider(0, 50, value=9.5, step=0.1, label="Guidance scale"),
                            gr.Slider(0, 1, step=0.01, label="DDIM sampling ETA"),
                            gr.Slider(1, 2, value=1, step=1, label="U-Net batch size"),
                            gr.Radio(["cuda", "cpu"], value="cuda", label="Device"),
                            gr.Text(label="Seed"),
                            gr.Text(value=args.outputs_path, label="Outputs path"),
                            gr.Radio(["png", "jpg"], value='png', label="Image format"),
                            gr.Checkbox(value=True, label="Turbo mode (better leave this on)"),
                            gr.Checkbox(label="Full precision mode (practically does nothing)"),
                            gr.Radio(["ddim", "plms", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"], value="plms", label="Sampler"),
                            gr.Slider(1, 10, value=1, step=1, label="speed_mp multiplier (don't change if not sure)"),
                        ], outputs=outs1)
                        b2.click(get_logs, inputs=[], outputs=outs2)
                        b3.click(get_nvidia_smi, inputs=[], outputs=outs3)

    demo.launch(share=True)
