import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import math
import time
import hashlib
import yaml
import argparse

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageEnhance
import torch.nn.functional as F
import torchvision


# for distributed evaluation
import torch.distributed as dist
from torch.multiprocessing import spawn
# for metrics
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from tools.fid_score import calculate_frechet_distance
from tools.inception import InceptionV3
import lpips
import warnings

warnings.filterwarnings("ignore")
from infinity.models.bsq_vae.vae import vae_model
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _pil_interp(method):
    if method == 'bicubic':
        return InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return InterpolationMode.LANCZOS
    elif method == 'hamming':
        return InterpolationMode.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return InterpolationMode.BILINEAR

def vae_encode_decode_norm(vae, image_path, tgt_h, tgt_w, device, augmentations):
    # get normalized gt_img and recons_img in [-1, 1]
    pil_image = Image.open(image_path).convert('RGB')
    # inp = crop_to_tensor(pil_image, tgt_h, tgt_w)
    inp = augmentations(pil_image)
    inp = inp * 2 - 1

    inp = inp.unsqueeze(0).to(device)

    # decode by vae
    # Both inputs and outputs are in [-1, 1]
    recons_img, vq_output = vae(inp)
    gt_img = inp

    return gt_img, recons_img # (1, 3, H, W)

def inference_eval(rank, world_size, args, vae, return_dict, val_txt, tgt_h, tgt_w, augmentations):
    # Don't remove this setup!!! dist.init_process_group is important for building loader (data.distributed.DistributedSampler)
    setup(rank, world_size) 

    device = torch.device(f"cuda:{rank}")

    for param in vae.parameters():
        param.requires_grad = False
    vae.to(device).eval()

    save_dir = 'results/%s'%(args.save)
    print('generating and saving video to %s...'%save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # data = VideoData(args)

    # loader = data.val_dataloader()

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    # loader_iter = iter(loader)

    pred_xs = []
    pred_recs = []
    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0
    
    # num_images = len(loader)
    assert len(val_txt) % world_size == 0
    num_images = len(val_txt) // world_size
    start_idx, end_idx = num_images * rank, num_images * (rank + 1)
    print(f"Testing {num_images} files")
    num_iter = 0

    for idx in tqdm(range(start_idx, end_idx)):
        rel_path = val_txt[idx]
        image_path = rel_path
        with torch.no_grad():
            torch.cuda.empty_cache()
            # x: [-1, 1]
            # x_recons, vq_output = vae(x.to(device), 2, 0, is_train=False)
            # x_recons = x_recons.cpu()
            x, x_recons = vae_encode_decode_norm(vae, image_path, tgt_h, tgt_w, device, augmentations)
            x_recons = x_recons.cpu()
        
        # paths = batch["path"]
        # assert len(paths) == x.shape[0]
        paths = [rel_path]

        for p, input_, recon_ in zip(paths, x, x_recons):
            if os.path.isabs(p):
                p = "/".join(p.split("/")[6:])
            assert not os.path.isabs(p), f"{p} should not be abspath"
            path = os.path.join(save_dir, "input_recon", os.path.basename(p))
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            input_ = ((input_ + 1) / 2).unsqueeze(0).to(device) # [-1, 1] -> [0, 1]
            
            pred_x = inception_model(input_)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()

            recon_ = ((recon_ + 1) / 2).unsqueeze(0).to(device) # [-1, 1] -> [0, 1]
            # recon_ = recon_.permute(1, 2, 0).detach().cpu()
            with torch.no_grad():
                pred_rec = inception_model(recon_)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()
            if args.save_prediction:
                if input_.dim() == 4:
                    input_image = input_.squeeze(0)
                if recon_.dim() == 4:
                    recon_image = recon_.squeeze(0)
                input_recon = torch.cat([input_image, recon_image], dim=-1)
                input_recon = Image.fromarray((torch.clamp(input_recon.permute(1, 2, 0).detach().cpu(), 0, 1).numpy() * 255).astype(np.uint8))
                input_recon.save(path)

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            # calculate lpips
            with torch.no_grad():
                lpips_alex += loss_fn_alex(input_, recon_, normalize=True).sum()
                lpips_vgg += loss_fn_vgg(input_, recon_, normalize=True).sum()

            #calculate PSNR and SSIM
            rgb_restored = (recon_ * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (input_ * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            ssim_temp = 0
            psnr_temp = 0
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                with torch.no_grad():
                    ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                    psnr_temp += psnr_loss(rgb_gt, rgb_restored)
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1
        
    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)
    temp_dict = {
        'pred_xs':pred_xs,
        'pred_recs':pred_recs,
        'lpips_alex':lpips_alex.cpu(),
        'lpips_vgg':lpips_vgg.cpu(),
        'ssim_value': ssim_value,
        'psnr_value': psnr_value,
        'num_iter': num_iter,
    }
    return_dict[rank] = temp_dict

    if dist.is_initialized():
        dist.barrier()
    cleanup()

def image_eval(pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter):
    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)
    
    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    lpips_alex_value = lpips_alex / num_iter
    lpips_vgg_value = lpips_vgg / num_iter
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter


    result_str = f"""
    FID = {fid_value:.4f}
    LPIPS_VGG: {lpips_vgg_value.item():.4f}
    LPIPS_ALEX: {lpips_alex_value.item():.4f}
    SSIM: {ssim_value:.4f}
    PSNR: {psnr_value:.3f}
    """
    return result_str

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_ckpt', type=str, default="infinity_vae_d32.pth")
    parser.add_argument('--codebook_dim', type=int, default=32)
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--save', type=str, default='imageNet_val')
    parser.add_argument('--tgt_size', type=int, default=256, help="input size during inference")
    parser.add_argument('--image_path', type=str, default=None) # "data/infinity_toy_data/images/5134521536907147208.jpg"
    args = parser.parse_args()
    return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args()
    
    # load bsq vae
    vqgan_ckpt = args.vqgan_ckpt
    schedule_mode = "dynamic"
    codebook_dim = args.codebook_dim
    codebook_size = 2**codebook_dim
    vae = vae_model(vqgan_ckpt, schedule_mode, codebook_dim, codebook_size)
    vae.eval()

    # read images
    if args.image_path is not None: # read a single image
        val_txt = [args.image_path]
        world_size = 1
    else: # test on benchmark
        val_txt_path = "data/labels/imagenet/val.txt"
        val_txt = open(val_txt_path, 'r').readlines()
        val_txt = [x.split("\t")[0] for x in val_txt if x.strip()] 
        world_size = torch.cuda.device_count()

    tgt_h, tgt_w = args.tgt_size, args.tgt_size
    resolution = (tgt_h, tgt_w)
    augmentations = transforms.Compose([
            transforms.Resize(min(resolution), interpolation=_pil_interp("bicubic")),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
    # get evaluation metrics
    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()

    spawn(inference_eval, args=(world_size, args, vae, return_dict, val_txt, tgt_h, tgt_w, augmentations), nprocs=world_size, join=True)

    pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter = [], [], 0, 0, 0, 0, 0
    for rank in range(world_size):
        pred_xs.append(return_dict[rank]['pred_xs'])
        pred_recs.append(return_dict[rank]['pred_recs'])
        lpips_alex += return_dict[rank]['lpips_alex']
        lpips_vgg += return_dict[rank]['lpips_vgg']
        ssim_value += return_dict[rank]['ssim_value']
        psnr_value += return_dict[rank]['psnr_value']
        num_iter += return_dict[rank]['num_iter']
    pred_xs = np.concatenate(pred_xs, 0)
    pred_recs = np.concatenate(pred_recs, 0)
    result_str = image_eval(pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter)
    print(result_str)