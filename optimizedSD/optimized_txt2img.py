import argparse
from datetime import datetime
import os
import random
import re
import time
from contextlib import nullcontext
from itertools import islice
from random import randint

import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging

from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger

logging.set_verbosity_error()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(image, h0, w0):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def get_image(opt, model, modelCS, modelFS, prompt=None, save=True, callback_fn=None):
    tic = time.time()
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.num_images, opt.C, opt.height // opt.f, opt.width // opt.f], device=opt.device)

    use_init_img = False
    try:
        init_image = load_img(opt.init_image, opt.height, opt.width).half().to(opt.device)
        use_init_img = True
    except:
        pass
    try:
        opt.seed
    except:
        opt.seed = randint(0, 1000000)

    speed_mp = opt.speed_mp
    batch_size = opt.num_images
    if not opt.from_file and prompt is None:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))

    if opt.precision == "autocast" and opt.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    if use_init_img:
        modelFS.to(opt.device)
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))
        z_enc = model.stochastic_encode(
            init_latent,
            torch.tensor([int(opt.ddim_steps * opt.img2img_strength)] * batch_size, device=opt.device),
            opt.seed,
            opt.ddim_eta,
            opt.ddim_steps,
        ).to(opt.device)
        if opt.device != "cpu":
            mem = torch.cuda.memory_allocated() / 1e6
            modelFS.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

    seeds = ""
    try:
        negative_prompt = opt.negative_prompt
    except:
        negative_prompt = ""
    with torch.no_grad():
        all_samples = list()
        count_iter = 0
        for _ in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                sample_path = os.path.join(opt.outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                if save:
                    os.makedirs(sample_path, exist_ok=True)
                    base_count = len(os.listdir(sample_path))
                else:
                    base_count = 0
                with precision_scope("cuda"):
                    modelCS.to(opt.device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [negative_prompt])
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

                    shape = [opt.num_images, opt.C, opt.height // opt.f, opt.width // opt.f]
                    seed_at_begin = opt.seed
                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    samples_ddim = model.sample(
                        x0=(z_enc if opt.sampler == "ddim" else init_latent) if use_init_img else None,
                        batch_size=batch_size,
                        S=opt.ddim_steps,
                        conditioning=c,
                        seed=opt.seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=start_code,
                        sampler=opt.sampler,
                        speed_mp=speed_mp,
                        callback_fn=callback_fn
                    )
                    modelFS.to(opt.device)

                    print(samples_ddim.shape)
                    print("saving images")
                    for i in range(batch_size):
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        seeds += str(opt.seed) + ","
                        opt.seed += 1
                        base_count += 1

                    count = 0
                    for x_index in range(count_iter * batch_size, len(all_samples)):
                        x_sample = 255. * rearrange(all_samples[x_index][0].numpy(), 'c h w -> h w c')
                        now = datetime.now()
                        dt_string = now.strftime("%d-%m-%Y %H_%M_%S ")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                                                            os.path.join(sample_path, f"{dt_string}s{seed_at_begin}d{opt.ddim_steps}-{str(count)}.png"))
                        count += 1
                        
                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)
            count_iter += 1
    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(
        (
                "Samples finished in {0:.2f} minutes"
        ).format(time_taken)
    )
    return all_samples
    # return Image.fromarray(grid.astype(np.uint8))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--config_path", type=str, default="optimizedSD/v1-inference.yaml",
        help="config path"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="checkpoint path"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="specify GPU (cuda/cuda:0/cuda:1/...)",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--unet_bs",
        type=int,
        default=1,
        help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
    )
    parser.add_argument(
        "--speed_mp",
        type=int,
        default=3,
        help="More vram, more image res (better not touch this )",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--format",
        type=str,
        help="output image format",
        choices=["jpg", "png"],
        default="png",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="sampler",
        choices=["ddim", "plms"],
        default="plms",
    )
    opt = parser.parse_args()
    opt.num_images = opt.n_samples
    opt.height = opt.H
    opt.width = opt.W

    tic = time.time()
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    opt.outpath = outpath
    grid_count = len(os.listdir(outpath)) - 1

    if opt.seed is None:
        opt.seed = randint(0, 1000000)
    seed_everything(opt.seed)

    # Logging
    logger(vars(opt), log_csv="logs/txt2img_logs.csv")

    sd = load_model_from_config(f"{opt.ckpt_path}")
    li, lo = [], []
    for key, value in sd.items():
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

    config = OmegaConf.load(f"{opt.config_path}")

    _model = instantiate_from_config(config.modelUNet)
    _, _ = _model.load_state_dict(sd, strict=False)
    _model.eval()
    _model.unet_bs = opt.unet_bs
    _model.cdevice = opt.device
    _model.turbo = opt.turbo

    _modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = _modelCS.load_state_dict(sd, strict=False)
    _modelCS.eval()
    _modelCS.cond_stage_model.device = opt.device

    _modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = _modelFS.load_state_dict(sd, strict=False)
    _modelFS.eval()
    del sd

    if opt.device != "cpu" and opt.precision == "autocast":
        _model.half()
        _modelCS.half()
        _modelFS.half()

    all_samples = get_image(
        opt,
        _model,
        _modelCS,
        _modelFS
    )

    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=opt.n_iter)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    
    now = datetime.now()
    dt_string = now.strftime(" %d-%m-%Y %H_%M_%S ")
    Image.fromarray(grid.astype(np.uint8)).save(
        os.path.join(outpath + "/" + str(opt.prompt).replace("/", "")[:100] + dt_string + f".{opt.format}")
    )
    print("exported to", outpath + "/" + str(opt.prompt).replace("/", "")[:100] + f".{opt.format}")
