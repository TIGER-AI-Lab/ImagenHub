# rewritten from https://github.com/DreamEditBenchTeam/DreamEdit/blob/main/src/generate_new.py
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from numpy import asarray
import random
from PIL import Image
from einops import rearrange, repeat

from tqdm.auto import tqdm
import math
import sys

from typing import List, Any

from .ldm.util import instantiate_from_config
from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.models.diffusion.plms import PLMSSampler
from .ldm.models.diffusion.dpm_solver import DPMSolverSampler
from .ldm.models.diffusion.dpm_solver import model_wrapper, NoiseScheduleVP, DPM_Solver

from diffusers import DDIMScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def load_model_from_pretrain(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    model = StableDiffusionPipeline.from_pretrained(
        ckpt,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            ckpt, subfolder="scheduler"
        ),
        torch_dtype=torch.float16,
    ).to("cuda")

    # model.eval()
    return model


def load_img(image_path, W, H):
    """
    modified to support PIL image as load
    """
    if type(image_path) is str:
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    #     w, h = image.size
    #     print(f"loaded input image of size ({w}, {h}) from {path}")
    #     w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((W, H), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def latent_to_image(model, latents):
    x_samples = model.decode_first_stage(latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
    x_samples = 255.0 * x_samples
    x_samples = x_samples.astype(np.uint8)

    return x_samples


def repeat_tensor(x, n, dim=0):
    dims = len(x.shape) * [1]
    dims[dim] = n
    return x.repeat(dims)


def get_mask(model, src, dst, init_latent, n: int, ddim_steps, clamp_rate: float = 1.5):
    """
    the map value will be clamped to map.mean() * clamp_rate, then values will be scaled into 0~1,
    then term into binary(split at 0.5).
    so if a map value is larger than map.mean() * clamp_rate * 0.5 will be encoded to 1, less will be encoded to 0.
    so the larger clamp rate is, fewer pixes will be encoded to 1,
    the small clamp rate is, the more pixes will be encoded to 1.
    """
    device = model.device
    repeated = repeat_tensor(init_latent, n)
    src = repeat_tensor(src, n)
    dst = repeat_tensor(dst, n)
    noise = torch.randn(init_latent.shape, device=device)
    scheduler = DDIMScheduler(
        num_train_timesteps=model.num_timesteps, trained_betas=model.betas.cpu().numpy()
    )
    scheduler.set_timesteps(ddim_steps, device=device)
    noised = scheduler.add_noise(repeated, noise, scheduler.timesteps[ddim_steps // 2])

    t = scheduler.timesteps[ddim_steps // 2]
    t_ = torch.unsqueeze(t, dim=0).to(device)
    pre_src = model.apply_model(noised, t_, src)
    pre_dst = model.apply_model(noised, t_, dst)

    # consider to add smooth method
    subed = (pre_src - pre_dst).abs_().mean(dim=[0, 1])
    max_v = subed.mean() * clamp_rate
    mask = subed.clamp(0, max_v) / max_v

    def to_binary(pix):
        if pix > 0.5:
            return 1.0
        else:
            return 0.0

    mask = mask.cpu().apply_(to_binary).to(device)
    return mask


def get_mask_new(
        model, src, dst, init_latent, sam_latent, n: int, ddim_steps, clamp_rate: float = 3
):
    """
    the map value will be clamped to map.mean() * clamp_rate, then values will be scaled into 0~1,
    then term into binary(split at 0.5).
    so if a map value is larger than map.mean() * clamp_rate * 0.5 will be encoded to 1, less will be encoded to 0.
    so the larger clamp rate is, fewer pixes will be encoded to 1,
    the small clamp rate is, the more pixes will be encoded to 1.
    """

    device = model.device

    # consider to add smooth method
    # subed = (pre_src - pre_dst).abs_().mean(dim=[0, 1])
    subed = (sam_latent - init_latent).abs_().mean(dim=[0, 1])

    max_v = subed.mean() * clamp_rate
    mask = subed.clamp(0, max_v) / max_v

    def to_binary(pix):
        if pix > 0.3:
            return 1.0
        else:
            return 0.0

    mask = mask.cpu().apply_(to_binary).to(device)
    return mask


# modify sample method of dpm_solver
# code adopt from
# https://github.com/Xiang-cd/DiffEdit-stable-diffusion/blob/main/diffedit.ipynb
# https://github.com/LuChengTHU/dpm-solver/blob/fafa2fb855ed63b625c517b4761bcf8d84f20f4b/dpm_solver_pytorch.py#LL1047C5-L1047C15
# here we just record sample lantents and apply mask in sample process
def sample_edit(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpm_solver",
        atol=0.0078,
        rtol=0.05,
        record_list=None,
        mask=None,
        beta=None,
        noise_step=10,
):
    t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
    t_T = self.noise_schedule.T if t_start is None else t_start
    device = x.device
    if record_list is not None:
        assert len(record_list) == steps
    if method == "adaptive":
        with torch.no_grad():
            x = self.dpm_solver_adaptive(
                x,
                order=order,
                t_T=t_T,
                t_0=t_0,
                atol=atol,
                rtol=rtol,
                solver_type=solver_type,
            )
    elif method == "multistep":
        assert steps >= order
        timesteps = self.get_time_steps(
            skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device
        )
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [self.model_fn(x, vec_t)]
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = self.multistep_dpm_solver_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    vec_t,
                    init_order,
                    solver_type=solver_type,
                )
                # TODO: not sure if this is necessary
                if noise_step > 0:
                    noise = torch.randn(x.shape, device=device)
                    scheduler = DDIMScheduler(
                        num_train_timesteps=noise_step, trained_betas=beta
                    )
                    scheduler.set_timesteps(noise_step, device=device)
                    x = scheduler.add_noise(
                        x, noise, scheduler.timesteps[noise_step // 2]
                    )
                if mask is not None and record_list is not None:
                    x = record_list[init_order - 1].to(device) * (1.0 - mask) + x * mask
                model_prev_list.append(self.model_fn(x, vec_t))
                t_prev_list.append(vec_t)
            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(order, steps + 1):
                vec_t = timesteps[step].expand(x.shape[0])
                if lower_order_final and steps < 15:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                x = self.multistep_dpm_solver_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    vec_t,
                    step_order,
                    solver_type=solver_type,
                )
                if mask is not None and record_list is not None:
                    x = record_list[step - 1].to(device) * (1.0 - mask) + x * mask
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = vec_t
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, vec_t)
    elif method in ["singlestep", "singlestep_fixed"]:
        if method == "singlestep":
            (
                timesteps_outer,
                orders,
            ) = self.get_orders_and_timesteps_for_singlestep_solver(
                steps=steps,
                order=order,
                skip_type=skip_type,
                t_T=t_T,
                t_0=t_0,
                device=device,
            )
        elif method == "singlestep_fixed":
            K = steps // order
            orders = [order, ] * K
            timesteps_outer = self.get_time_steps(
                skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device
            )
        for i, order in enumerate(orders):
            t_T_inner, t_0_inner = timesteps_outer[i], timesteps_outer[i + 1]
            timesteps_inner = self.get_time_steps(
                skip_type=skip_type,
                t_T=t_T_inner.item(),
                t_0=t_0_inner.item(),
                N=order,
                device=device,
            )
            lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
            vec_s, vec_t = t_T_inner.tile(x.shape[0]), t_0_inner.tile(x.shape[0])
            h = lambda_inner[-1] - lambda_inner[0]
            r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
            r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
            x = self.singlestep_dpm_solver_update(
                x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2
            )
    if denoise_to_zero:
        x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
    return x


def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpm_solver",
        atol=0.0078,
        rtol=0.05,
        record_process=False,
        record_list=None,
):
    t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
    t_T = self.noise_schedule.T if t_start is None else t_start
    device = x.device
    if method == "adaptive":
        with torch.no_grad():
            x = self.dpm_solver_adaptive(
                x,
                order=order,
                t_T=t_T,
                t_0=t_0,
                atol=atol,
                rtol=rtol,
                solver_type=solver_type,
            )
    elif method == "multistep":
        assert steps >= order
        timesteps = self.get_time_steps(
            skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device
        )
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [self.model_fn(x, vec_t)]
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = self.multistep_dpm_solver_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    vec_t,
                    init_order,
                    solver_type=solver_type,
                )
                if record_process:
                    record_list.append(x.cpu())
                model_prev_list.append(self.model_fn(x, vec_t))
                t_prev_list.append(vec_t)
            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(order, steps + 1):
                vec_t = timesteps[step].expand(x.shape[0])
                if lower_order_final and steps < 15:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                x = self.multistep_dpm_solver_update(
                    x,
                    model_prev_list,
                    t_prev_list,
                    vec_t,
                    step_order,
                    solver_type=solver_type,
                )
                if record_process:
                    record_list.append(x.cpu())
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = vec_t
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, vec_t)
    elif method in ["singlestep", "singlestep_fixed"]:
        if method == "singlestep":
            (
                timesteps_outer,
                orders,
            ) = self.get_orders_and_timesteps_for_singlestep_solver(
                steps=steps,
                order=order,
                skip_type=skip_type,
                t_T=t_T,
                t_0=t_0,
                device=device,
            )
        elif method == "singlestep_fixed":
            K = steps // order
            orders = [order, ] * K
            timesteps_outer = self.get_time_steps(
                skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device
            )
        for i, order in enumerate(orders):
            t_T_inner, t_0_inner = timesteps_outer[i], timesteps_outer[i + 1]
            timesteps_inner = self.get_time_steps(
                skip_type=skip_type,
                t_T=t_T_inner.item(),
                t_0=t_0_inner.item(),
                N=order,
                device=device,
            )
            lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
            vec_s, vec_t = t_T_inner.tile(x.shape[0]), t_0_inner.tile(x.shape[0])
            h = lambda_inner[-1] - lambda_inner[0]
            r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
            r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
            x = self.singlestep_dpm_solver_update(
                x, vec_s, vec_t, order, solver_type=solver_type, r1=r1, r2=r2
            )
    if denoise_to_zero:
        x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
    return x


def diffedit(
        model,
        config,
        init_image,
        iteration: int,
        src_prompt: str = "photo of a dog with background",
        dst_prompt: str = "photo of a gdj dog with background",
        encode_ratio: float = 0.6,
        ddim_steps: int = 40,
        seed: int = 42,
        scale: float = 7.5,
        precision: str = "autocast",
        sam_mask: torch.Tensor = None,
        W: int = 0,
        H: int = 0,
        noise_step: int = 10,
        record_list: List[Any] = None,
        use_diffedit: bool = False
):
    """
    :param init_image: image to be edit
    :param src_prompt: prompt describe origin image(i.e. A bowl of fruits)
    :param dst_prompt: prompt describe desired image(i.e. A bowl of pears)
    :param encode_ratio: how deep to encode origin image, must between 0-1
    :param ddim_steps: total ddim steps, actual encode steps = ddim_steps * encode ratio
    :param seed: random seed
    :param scale: classifier free guidance scale
    :param precision: ema precision
    """
    # If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None:
        seed = random.randrange(2 ** 32 - 1)
    seed_everything(seed)
    device = model.device

    model.cond_stage_model = model.cond_stage_model.to(device)
    precision_scope = autocast if precision == "autocast" else nullcontext
    # assert os.path.isfile(opt.origin_image)
    init_image = load_img(init_image, W, H).to(device)
    init_image = repeat(init_image, "1 ... -> b ...", b=1)

    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning([""])
                src = model.get_learned_conditioning([src_prompt])
                dst = model.get_learned_conditioning([dst_prompt])
                init_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(init_image)
                )
                if use_diffedit:
                    sam_mask = get_mask(model, src, dst, init_latent, 3, ddim_steps)
                ns = NoiseScheduleVP("discrete", betas=model.betas)
                model_fn = model_wrapper(
                    lambda x, t, c: model.apply_model(x, t, c),
                    ns,
                    model_type="noise",
                    guidance_type="classifier-free",
                    condition=src,
                    unconditional_condition=uc,
                    guidance_scale=scale,
                )

                # add noise and record each step's output latent
                noiser = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
                noiser.sample = sample.__get__(noiser, type(noiser))

                if iteration != 0 and config.background_correction.use_latents_record and record_list is not None:
                    noised_sample = noiser.sample(
                        init_latent,
                        t_start=1.0 / model.num_timesteps,
                        t_end=encode_ratio,
                        method="multistep",
                        order=2,
                        steps=ddim_steps,
                        record_process=True,
                        record_list=[],
                    )
                else:
                    record_list = []
                    noised_sample = noiser.sample(
                        init_latent,
                        t_start=1.0 / model.num_timesteps,
                        t_end=encode_ratio,
                        method="multistep",
                        order=2,
                        steps=ddim_steps,
                        record_process=True,
                        record_list=record_list,
                    )
                    assert not record_list == []

                # perform step wise edit
                model_fn_dst = model_wrapper(
                    lambda x, t, c: model.apply_model(x, t, c),
                    ns,
                    model_type="noise",
                    guidance_type="classifier-free",
                    condition=dst,
                    unconditional_condition=uc,
                    guidance_scale=scale,
                )
                solver = DPM_Solver(
                    model_fn_dst, ns, predict_x0=True, thresholding=False
                )

                solver.sample_edit = sample_edit.__get__(solver, type(solver))
                recover = solver.sample_edit(
                    noised_sample,
                    t_start=encode_ratio,
                    t_end=1.0 / model.num_timesteps,
                    method="multistep",
                    order=2,
                    steps=ddim_steps,
                    mask=sam_mask,
                    record_list=list(reversed(record_list)),
                    beta=model.betas.cpu().numpy(),
                    noise_step=noise_step,
                )

                images = latent_to_image(model, recover)
                org_images = latent_to_image(model, init_latent)

                if config.background_correction.use_latents_record:
                    if use_diffedit:
                        return images, org_images, record_list, sam_mask
                    else:
                        return images, org_images, record_list, None
                else:
                    if use_diffedit:
                        return images, org_images, [], sam_mask
                    else:
                        return images, org_images, [], None
