import os
import sys

import cv2
import git

if not os.path.exists("CodeFormer/"):
    print("Installing CodeFormer..")
    git.Repo.clone_from("https://github.com/sczhou/CodeFormer/", "CodeFormer")
    os.chdir("CodeFormer")
    os.system("python basicsr/setup.py develop")
    os.chdir("..")
    print("Installation successful")

sys.path.append('CodeFormer/')
sys.path.append('../CodeFormer/')

import argparse
import asyncio
import logging
import mimetypes
import re
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
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging as transformers_logging

from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer

from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.transforms.functional import normalize

transformers_logging.set_verbosity_error()

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


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
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def load_mask(mask, h0, w0, newH, newW, invert=False):
    image = mask.convert("RGB")
    w, h = image.size
    print(f"loaded input mask of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New mask size ({w}, {h})")
    image = image.resize((newW, newH), resample=Image.LANCZOS)
    # image = image.resize((64, 64), resample=Image.LANCZOS)
    image = np.array(image)

    if invert:
        print("inverted")
        where_0, where_1 = np.where(image == 0), np.where(image == 255)
        image[where_0], image[where_1] = 255, 0
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def toImgOpenCV(imgPIL):  # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL)  # After mapping from PIL to numpy : [R,G,B,A]
    # numpy Image Channel system: [B,G,R,A]
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i


async def get_logs():
    return "\n".join([x for x in open("log.txt", "r", encoding="utf8").readlines()] +
                     [y for y in open("tqdm.txt", "r", encoding="utf8").readlines()])


async def get_nvidia_smi():
    proc = await asyncio.create_subprocess_shell('nvidia-smi', stdout=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return str(stdout)


def generate_img2img(
        image,
        prompt,
        strength,
        ddim_steps,
        n_iter,
        batch_size,
        Width,
        Height,
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
        speed_mp,
):
    logging.info(f"prompt: {prompt}, W: {Width}, H: {Height}")
    try:
        init_image = load_img(image['image'], Height, Width).to(device)
    except:
        init_image = load_img(image, Height, Width).to(device)
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    try:
        seed = int(seed)
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
    try:
        image['mask']
        use_mask = True
    except:
        use_mask = False
    if use_mask:
        mask = load_mask(image['mask'], Height, Width, init_latent.shape[2], init_latent.shape[3], True).to(device)
        mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
        if device != "cpu" and not full_precision:
            mask = mask.half().to(device)

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
                        batch_size=batch_size,
                        x_T=init_latent if use_mask else None,
                        mask=mask if use_mask else None
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


def generate_img2img_interp(
        image,
        prompt,
        strength,
        ddim_steps,
        n_iter,
        batch_size,
        Width,
        Height,
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
        speed_mp,
        n_interpolate_samples
):
    logging.info(f"prompt: {prompt}, W: {Width}, H: {Height}")
    try:
        init_image = load_img(image['image'], Height, Width).to(device)
    except:
        init_image = load_img(image, Height, Width).to(device)
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    try:
        seed = int(seed)
    except:
        seed = randint(0, 1000000)

    if device != "cpu" and not full_precision:
        model.half()
        modelCS.half()
        modelFS.half()
        init_image = init_image.half()

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
                    true_z_enc = model.stochastic_encode(
                        init_latent, torch.tensor([t_enc] * batch_size).to(device), seed, ddim_eta, ddim_steps
                    )
                    # decode it
                    samples_ddim = model.sample(
                        t_enc,
                        c,
                        true_z_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        sampler=sampler,
                        speed_mp=speed_mp
                    )
                    modelFS.to(device)
                    print("decoding frames")
                    all_time_samples = []
                    for ij in tqdm(range(n_interpolate_samples)):
                        temp_all_samples = []
                        for i in range(batch_size):
                            start0_sample = samples_ddim[i].unsqueeze(0)
                            interp_sample = torch.lerp(init_latent, start0_sample, (ij / n_interpolate_samples))
                            x_samples_ddim = modelFS.decode_first_stage(interp_sample)
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            temp_all_samples.append(x_sample.to("cpu"))
                            x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{img_format}")
                            )
                            seeds += str(seed) + ","
                            base_count += 1
                        grid = torch.cat(temp_all_samples, 0)
                        grid = make_grid(grid, nrow=n_iter)
                        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                        all_time_samples.append(Image.fromarray(grid.astype(np.uint8)))
                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)
        # all_samples.append(all_time_samples)
    print("creating a video..")
    all_time_samples = [toImgOpenCV(img) for img in all_time_samples]
    out = cv2.VideoWriter("tempfile.mp4", cv2.VideoWriter_fourcc(*'h264'), 15, (all_time_samples[0].shape[1], all_time_samples[0].shape[0]))
    for img in all_time_samples:
        out.write(img)
    out.release()

    return "tempfile.mp4", f"yeah here's your video {Width}x{Height}"


def generate_double_triple(
        prompt,
        ddim_steps,
        img2img_strength,
        Width,
        Height,
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
        speed_mp,
        upscale_reso
):
    C = 4
    f = 8
    start_code = None
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)

    if device != "cpu" and not full_precision:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt.replace("/", ""))))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [1 * [prompt]]

    if device != "cpu" and not full_precision:
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    with torch.no_grad():
        all_samples = list()
        for prompts in tqdm(data, desc="data"):
            with precision_scope("cuda"):
                modelCS.to(device)
                uc = None
                if scale != 1.0:
                    uc = modelCS.get_learned_conditioning(1 * [""])
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

                shape = [1, C, Height // f, Width // f]

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                samples_ddim = model.sample(
                    S=ddim_steps,
                    conditioning=c,
                    seed=seed,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    x_T=start_code,
                    sampler=sampler,
                    speed_mp=speed_mp
                )

                modelFS.to(device)

                x_samples_ddim = modelFS.decode_first_stage(samples_ddim[0].unsqueeze(0))
                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                Image.fromarray(x_sample.astype(np.uint8)).save(
                    os.path.join(sample_path, "seed_" + str(seed) + "_step1_" + f"{base_count:05}.{img_format}")
                )

                ### STEP 2

                init_image = repeat(
                    load_img(Image.fromarray(x_sample.astype(np.uint8)), Height * 2, Width * 2).to(device),
                    "1 ... -> b ...", b=1)
                init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))

                modelFS.cpu()
                model.to(device)

                z_enc = model.stochastic_encode(
                    init_latent, torch.tensor([int(img2img_strength * ddim_steps)]).to(device), seed, ddim_eta,
                    ddim_steps
                ).to(device)

                samples_ddim = model.sample(
                    int(img2img_strength * ddim_steps),
                    c,
                    z_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    sampler="ddim",
                    speed_mp=speed_mp
                )

                modelFS.to(device)
                model.cpu()
                modelCS.to("cpu")
                torch.cuda.empty_cache()

                x_samples_ddim = modelFS.decode_first_stage(samples_ddim[0].unsqueeze(0))
                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                if upscale_reso < 3:
                    all_samples.append(x_sample.cpu())
                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                Image.fromarray(x_sample.astype(np.uint8)).save(
                    os.path.join(sample_path, "seed_" + str(seed) + "_step2_" + f"{base_count:05}.{img_format}")
                )

                ### STEP 3
                if upscale_reso >= 3:
                    init_image = repeat(
                        load_img(Image.fromarray(x_sample.astype(np.uint8)), Height * 3, Width * 3).to(device),
                        "1 ... -> b ...", b=1)
                    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))

                    modelFS.cpu()
                    model.to(device)

                    z_enc = model.stochastic_encode(
                        init_latent, torch.tensor([int(img2img_strength * ddim_steps)]).to(device), seed, ddim_eta,
                        ddim_steps
                    ).to(device)

                    samples_ddim = model.sample(
                        int(img2img_strength * ddim_steps),
                        c,
                        z_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        sampler="ddim",
                        speed_mp=speed_mp
                    )

                    print("saving images")
                    model.cpu()
                    modelFS.to(device)

                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[0].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    all_samples.append(x_sample.to("cpu"))
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, "seed_" + str(seed) + "_step3_" + f"{base_count:05}.{img_format}")
                    )

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
    grid = make_grid(grid, nrow=1)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    txt = (
            "Samples finished in "
            + str(round(time_taken, 3))
            + " minutes and exported to "
            + sample_path
            + "\nSeeds used = "
            + seeds[:-1]
    )
    return Image.fromarray(grid.astype(np.uint8)), txt


def upscale2x(img):
    img = Image.fromarray(upsampler.enhance(img, outscale=2)[0])
    return img, f"Upscaled to resolution: {img.size}"


def face_restore(img):
    if img is None:
        print("Image is not ready!")
        return None
    only_center_face = False
    draw_box = False
    codeformer_fidelity = 0.5
    upscale = 2
    face_upsample = True
    detection_model = "retinaface_resnet50"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_helper = FaceRestoreHelper(
        True,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )
    codeformer_net.to(device)
    bg_upsampler = upsampler
    face_upsampler = upsampler
    face_helper.read_image(img)
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=only_center_face, resize=640, eye_dist_threshold=5
    )
    print(f"\tdetect {num_det_faces} faces")
    # align and warp each face
    face_helper.align_warp_face()

    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(
            cropped_face / 255.0, bgr2rgb=True, float32=True
        )
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = codeformer_net(
                    cropped_face_t, w=codeformer_fidelity, adain=True
                )[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f"\tFailed inference for CodeFormer: {error}")
            restored_face = tensor2img(
                cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
            )

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    # paste_back
    # upsample the background
    if bg_upsampler is not None:
        # Now only support RealESRGAN for upsampling background
        bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
    else:
        bg_img = None
    face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    if face_upsample and face_upsampler is not None:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=draw_box,
            face_upsampler=face_upsampler,
        )
    else:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img, draw_box=draw_box
        )
    img = Image.fromarray(restored_img)
    return img, f"Fixed a face, new img size: {img.size}"


def generate_txt2img(
        prompt,
        ddim_steps,
        n_iter,
        batch_size,
        Width,
        Height,
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
    logging.info(f"prompt: {prompt}, W: {Width}, H: {Height}")
    C = 4
    f = 8
    start_code = None
    model.to(device)
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)

    if device != "cpu" and not full_precision:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt.replace("/", ""))))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    if device != "cpu" and not full_precision:
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    with torch.no_grad():
        all_samples = list()
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

                    shape = [batch_size, C, Height // f, Width // f]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler=sampler,
                        speed_mp=speed_mp
                    )

                    modelFS.to(device)
                    model.cpu()
                    logging.info("saving images")
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
                    logging.info(str("memory_final = " + str(torch.cuda.memory_allocated() / 1e6)))

    toc = time.time()

    time_taken = (toc - tic) / 60.0
    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=n_iter)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    txt = (
            "Samples finished in "
            + str(round(time_taken, 3))
            + " minutes and exported to "
            + sample_path
            + "\nSeeds used = "
            + seeds[:-1]
    )
    return Image.fromarray(grid.astype(np.uint8)), txt


def download_codeformer(args):
    pretrain_model_url = {
        'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
        'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
    }
    # download weights
    if not os.path.exists(args.codeformer_path + "CodeFormer/codeformer.pth"):
        load_file_from_url(url=pretrain_model_url['codeformer'], model_dir=args.codeformer_path + "CodeFormer",
                           progress=True, file_name=None)
    if not os.path.exists(args.codeformer_path + "facelib/detection_Resnet50_Final.pth"):
        load_file_from_url(url=pretrain_model_url['detection'], model_dir=args.codeformer_path + "facelib",
                           progress=True,
                           file_name=None)
    if not os.path.exists(args.codeformer_path + "facelib/parsing_parsenet.pth"):
        load_file_from_url(url=pretrain_model_url['parsing'], model_dir=args.codeformer_path + "facelib", progress=True,
                           file_name=None)
    if not os.path.exists(args.codeformer_path + "realesrgan/RealESRGAN_x2plus.pth"):
        load_file_from_url(url=pretrain_model_url['realesrgan'], model_dir=args.codeformer_path + "realesrgan",
                           progress=True, file_name=None)


# set enhancer with RealESRGAN
def set_realesrgan(args):
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path=args.codeformer_path + "realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


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
    global lines, use_mask
    use_mask = True  # by default is false
    lines = []
    file_handler = logging.FileHandler(filename='log.txt', mode='w')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler, TqdmLoggingHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    parser = argparse.ArgumentParser(description='SD by neonsecret using gradio')
    parser.add_argument('--config_path', default="optimizedSD/v1-inference.yaml", type=str, help='config path')
    parser.add_argument('--ckpt_path', default="models/ldm/stable-diffusion-v1/model.ckpt", type=str, help='ckpt path')
    parser.add_argument('--codeformer_path', default="models/codeformer/", type=str, help='ckpt path')
    parser.add_argument('--outputs_path', default="outputs/output-samples", type=str, help='output imgs path')
    args = parser.parse_args()
    args.codeformer_path = args.codeformer_path + "/" if args.codeformer_path[-1] != "/" else args.codeformer_path

    print("Downloading codeformer weights..")
    download_codeformer(args)

    print("Loading realesr..")
    upsampler = set_realesrgan(args)
    print("Loading codeformer..")
    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    )
    checkpoint = torch.load(args.codeformer_path + "CodeFormer/codeformer.pth")["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

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
        with gr.Tab("txt2img"):
            with gr.Column():
                gr.Markdown("# Generate images from text (neonsecret's adjustments)")
                gr.Markdown("### Press 'generation status' button to get the model output logs")
                with gr.Row():
                    with gr.Column():
                        out_image = gr.Image(label="Output Image")
                        gen_res = gr.Text(label="Generation results")
                        outs2 = [gr.Text(label="Logs")]
                        outs3 = gr.Text(label="nvidia-smi")
                        b1 = gr.Button("Generate!")
                        b4 = gr.Button("Face correction")
                        b5 = gr.Button("Upscale 2x")
                        b2 = gr.Button("generation status")
                        b3 = gr.Button("nvidia-smi")
                    with gr.Column():
                        with gr.Box():
                            b4.click(face_restore, inputs=[out_image], outputs=[out_image, gen_res])
                            b5.click(upscale2x, inputs=[out_image], outputs=[out_image, gen_res])
                            b1.click(generate_txt2img, inputs=[
                                gr.Text(label="Your Prompt"),
                                gr.Slider(1, 200, value=50, label="Sampling Steps"),
                                gr.Slider(1, 100, step=1, label="Number of images"),
                                gr.Slider(1, 100, step=1, label="Batch size"),
                                gr.Slider(64, 4096, value=512, step=64, label="Width"),
                                gr.Slider(64, 4096, value=512, step=64, label="Height"),
                                gr.Slider(-25, 25, value=7.5, step=0.1, label="Guidance scale"),
                                gr.Slider(0, 1, step=0.01, label="DDIM sampling ETA"),
                                gr.Slider(1, 2, value=1, step=1, label="U-Net batch size"),
                                gr.Radio(["cuda", "cpu"], value="cuda", label="Device"),
                                gr.Text(label="Seed"),
                                gr.Text(value=args.outputs_path, label="Outputs path"),
                                gr.Radio(["png", "jpg"], value='png', label="Image format"),
                                gr.Checkbox(value=True, label="Turbo mode (better leave this on)"),
                                gr.Checkbox(label="Full precision mode (practically does nothing)"),
                                gr.Radio(
                                    ["ddim", "plms", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"],
                                    value="plms", label="Sampler"),
                                gr.Slider(1, 100, value=100, step=1,
                                          label="%, VRAM usage limiter (100 means max speed)"),
                            ], outputs=[out_image, gen_res])
                            b2.click(get_logs, inputs=[], outputs=outs2)
                            b3.click(get_nvidia_smi, inputs=[], outputs=[outs3])
        with gr.Tab("img2img"):
            with gr.Column():
                gr.Markdown("# Generate images from images (neonsecret's adjustments)")
                gr.Markdown("### Press 'generation status' button to get the model output logs")
                with gr.Row():
                    with gr.Column():
                        out_image2 = gr.Image(label="Output Image")
                        gen_res2 = gr.Text(label="Generation results")
                        outs2 = [gr.Text(label="Logs")]
                        outs3 = [gr.Text(label="nvidia-smi")]
                        b1 = gr.Button("Generate!")
                        b4 = gr.Button("Face correction")
                        b5 = gr.Button("Upscale 2x")
                        b2 = gr.Button("generation status")
                        b3 = gr.Button("nvidia-smi")
                    with gr.Column():
                        with gr.Box():
                            b4.click(face_restore, inputs=[out_image2], outputs=[out_image2, gen_res2])
                            b5.click(upscale2x, inputs=[out_image2], outputs=[out_image2, gen_res2])
                            b1.click(generate_img2img, inputs=[
                                gr.Image(tool="editor", type="pil", label="Initial image"),
                                gr.Text(label="Your Prompt"),
                                gr.Slider(0, 1, value=0.75, label="Generated image strength"),
                                gr.Slider(1, 200, value=50, label="Sampling Steps"),
                                gr.Slider(1, 100, step=1, label="Number of images"),
                                gr.Slider(1, 100, step=1, label="Batch size"),
                                gr.Slider(64, 4096, value=512, step=64, label="Width"),
                                gr.Slider(64, 4096, value=512, step=64, label="Height"),
                                gr.Slider(-25, 25, value=7.5, step=0.1, label="Guidance scale"),
                                gr.Slider(0, 1, step=0.01, label="DDIM sampling ETA"),
                                gr.Slider(1, 2, value=1, step=1, label="U-Net batch size"),
                                gr.Radio(["cuda", "cpu"], value="cuda", label="Device"),
                                gr.Text(label="Seed"),
                                gr.Text(value=args.outputs_path, label="Outputs path"),
                                gr.Radio(["png", "jpg"], value='png', label="Image format"),
                                gr.Checkbox(value=True, label="Turbo mode (better leave this on)"),
                                gr.Checkbox(label="Full precision mode (practically does nothing)"),
                                gr.Radio(
                                    ["ddim", "plms", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"],
                                    value="ddim", label="Sampler"),
                                gr.Slider(1, 100, value=100, step=1,
                                          label="%, VRAM usage limiter (100 means max speed)"),
                            ], outputs=[out_image2, gen_res2])
                            b2.click(get_logs, inputs=[], outputs=outs2)
                            b3.click(get_nvidia_smi, inputs=[], outputs=outs3)
        with gr.Tab("img2img inpaint"):
            with gr.Column():
                gr.Markdown("# Generate images from images (with a mask) (neonsecret's adjustments)")
                gr.Markdown("### Press 'generation status' button to get the model output logs")
                with gr.Row():
                    with gr.Column():
                        out_image3 = gr.Image(label="Output Image")
                        gen_res3 = gr.Text(label="Generation results")
                        outs2 = [gr.Text(label="Logs")]
                        outs3 = [gr.Text(label="nvidia-smi")]
                        b1 = gr.Button("Generate!")
                        b4 = gr.Button("Face correction")
                        b5 = gr.Button("Upscale 2x")
                        b2 = gr.Button("generation status")
                        b3 = gr.Button("nvidia-smi")
                    with gr.Column():
                        with gr.Box():
                            b4.click(face_restore, inputs=[out_image3], outputs=[out_image3, gen_res3])
                            b5.click(upscale2x, inputs=[out_image3], outputs=[out_image3, gen_res3])
                            b1.click(generate_img2img, inputs=[
                                gr.Image(tool="sketch", type="pil", label="Initial image with a mask"),
                                gr.Text(label="Your Prompt"),
                                gr.Slider(0, 1, value=0.75, label="Generated image strength"),
                                gr.Slider(1, 200, value=50, label="Sampling Steps"),
                                gr.Slider(1, 100, step=1, label="Number of images"),
                                gr.Slider(1, 100, step=1, label="Batch size"),
                                gr.Slider(64, 4096, value=512, step=64, label="Width"),
                                gr.Slider(64, 4096, value=512, step=64, label="Height"),
                                gr.Slider(-25, 25, value=7.5, step=0.1, label="Guidance scale"),
                                gr.Slider(0, 1, step=0.01, label="DDIM sampling ETA"),
                                gr.Slider(1, 2, value=1, step=1, label="U-Net batch size"),
                                gr.Radio(["cuda", "cpu"], value="cuda", label="Device"),
                                gr.Text(label="Seed"),
                                gr.Text(value=args.outputs_path, label="Outputs path"),
                                gr.Radio(["png", "jpg"], value='png', label="Image format"),
                                gr.Checkbox(value=True, label="Turbo mode (better leave this on)"),
                                gr.Checkbox(label="Full precision mode (practically does nothing)"),
                                gr.Radio(
                                    ["ddim", "plms", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"],
                                    value="ddim", label="Sampler"),
                                gr.Slider(1, 100, value=100, step=1,
                                          label="%, VRAM usage limiter (100 means max speed)"),
                            ], outputs=[out_image3, gen_res3])
                            b2.click(get_logs, inputs=[], outputs=outs2)
                            b3.click(get_nvidia_smi, inputs=[], outputs=outs3)
        with gr.Tab("img2img interpolate"):
            with gr.Column():
                gr.Markdown("# Generate a video interpolation from images")
                gr.Markdown("### Press 'generation status' button to get the model output logs")
                with gr.Row():
                    with gr.Column():
                        out_video = gr.Video()
                        gen_res4 = gr.Text(label="Generation results")
                        outs2 = [gr.Text(label="Logs")]
                        outs3 = [gr.Text(label="nvidia-smi")]
                        b1 = gr.Button("Generate!")
                        b2 = gr.Button("generation status")
                        b3 = gr.Button("nvidia-smi")
                    with gr.Column():
                        with gr.Box():
                            b1.click(generate_img2img_interp, inputs=[
                                gr.Image(tool="editor", type="pil", label="Initial image"),
                                gr.Text(label="Your Prompt"),
                                gr.Slider(0, 1, value=0.75, label="Generated image strength"),
                                gr.Slider(1, 200, value=50, label="Sampling Steps"),
                                gr.Slider(1, 100, step=1, label="Number of images"),
                                gr.Slider(1, 100, step=1, label="Batch size"),
                                gr.Slider(64, 4096, value=512, step=64, label="Width"),
                                gr.Slider(64, 4096, value=512, step=64, label="Height"),
                                gr.Slider(-25, 25, value=7.5, step=0.1, label="Guidance scale"),
                                gr.Slider(0, 1, step=0.01, label="DDIM sampling ETA"),
                                gr.Slider(1, 2, value=1, step=1, label="U-Net batch size"),
                                gr.Radio(["cuda", "cpu"], value="cuda", label="Device"),
                                gr.Text(label="Seed"),
                                gr.Text(value=args.outputs_path, label="Outputs path"),
                                gr.Radio(["png", "jpg"], value='png', label="Image format"),
                                gr.Checkbox(value=True, label="Turbo mode (better leave this on)"),
                                gr.Checkbox(label="Full precision mode (practically does nothing)"),
                                gr.Radio(
                                    ["ddim", "plms", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"],
                                    value="ddim", label="Sampler"),
                                gr.Slider(1, 100, value=100, step=1,
                                          label="%, VRAM usage limiter (100 means max speed)"),
                                gr.Slider(1, 120, value=60, step=1, label="How smooth/slow the video will be"),
                            ], outputs=[out_video, gen_res4])
                            b2.click(get_logs, inputs=[], outputs=outs2)
                            b3.click(get_nvidia_smi, inputs=[], outputs=outs3)
        with gr.Tab("txt2img 2x-3x upscale"):
            with gr.Column():
                gr.Markdown("# Generate images from text using SD upscaling")
                gr.Markdown("### Generate images in 2(3) steps - Wx -> 2Wx2H (-> 3Wx3H)")
                gr.Markdown("### Press 'generation status' button to get the model output logs")
                with gr.Row():
                    with gr.Column():
                        out_image = gr.Image(label="Output Image")
                        gen_res = gr.Text(label="Generation results")
                        outs2 = [gr.Text(label="Logs")]
                        outs3 = gr.Text(label="nvidia-smi")
                        b1 = gr.Button("Generate!")
                        b4 = gr.Button("Face correction")
                        b5 = gr.Button("Upscale 2x")
                        b2 = gr.Button("generation status")
                        b3 = gr.Button("nvidia-smi")
                    with gr.Column():
                        with gr.Box():
                            b4.click(face_restore, inputs=[out_image], outputs=[out_image, gen_res])
                            b5.click(upscale2x, inputs=[out_image], outputs=[out_image, gen_res])
                            b1.click(generate_double_triple, inputs=[
                                gr.Text(label="Your Prompt"),
                                gr.Slider(1, 200, value=50, label="Sampling Steps"),
                                gr.Slider(0, 1, value=0.35, label="Upscaled image changes strength"),
                                gr.Slider(64, 4096, value=512, step=64, label="Width"),
                                gr.Slider(64, 4096, value=512, step=64, label="Height"),
                                gr.Slider(-25, 25, value=7.5, step=0.1, label="Guidance scale"),
                                gr.Slider(0, 1, step=0.01, label="DDIM sampling ETA"),
                                gr.Slider(1, 2, value=1, step=1, label="U-Net batch size"),
                                gr.Radio(["cuda", "cpu"], value="cuda", label="Device"),
                                gr.Text(label="Seed"),
                                gr.Text(value=args.outputs_path, label="Outputs path"),
                                gr.Radio(["png", "jpg"], value='png', label="Image format"),
                                gr.Checkbox(value=True, label="Turbo mode (better leave this on)"),
                                gr.Checkbox(label="Full precision mode (practically does nothing)"),
                                gr.Radio(
                                    ["ddim", "plms", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"],
                                    value="plms", label="Sampler"),
                                gr.Slider(1, 100, value=100, step=1,
                                          label="%, VRAM usage limiter (100 means max speed)"),
                                gr.Slider(2, 3, value=2, step=1,
                                          label="Neural scaling factor, 3 will take much longer"),
                            ], outputs=[out_image, gen_res])
                            b2.click(get_logs, inputs=[], outputs=outs2)
                            b3.click(get_nvidia_smi, inputs=[], outputs=[outs3])
    demo.launch(share=True)
