#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-02 20:43:41

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte
import lpips
import pickle

from utils import util_opts
from utils import util_image
from utils import util_common

from sampler import DifIRSampler
from ResizeRight.resize_right import resize
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import img2tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--gpu_id",
            type=str,
            default='0',
            help="GPU Index",
            )
    parser.add_argument(
            "-s",
            "--started_timesteps",
            type=int,
            default='100',
            help='Started timestep for DifFace, parameter N in our paper (Default:100)',
            )
    parser.add_argument(
            "--aligned",
            action='store_true',
            help='Input are alinged faces',
            )
    parser.add_argument(
            "--draw_box",
            action='store_true',
            help='Draw box for face in the unaligned case',
            )
    parser.add_argument(
            "-t",
            "--timestep_respacing",
            type=str,
            default='250',
            help='Sampling steps for Improved DDPM, parameter T in out paper (default 250)',
            )
    parser.add_argument(
            "--in_path",
            type=str,
            default='./testdata/cropped_faces',
            help='Folder to save the low quality image',
            )
    parser.add_argument(
            "--out_path",
            type=str,
            default='./results',
            help='Folder to save the restored results',
            )
    parser.add_argument(
            "--gt_path",
            type=str,
            default=None,
            help='Folder to GT images',
            )
    parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help='path to the ckpt to be tested',
            )
    args = parser.parse_args()

    cfg_path = 'configs/sample/iddpm_ffhq512_swinir.yaml'

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = args.gpu_id
    configs.aligned = args.aligned
    assert args.started_timesteps < int(args.timestep_respacing)
    configs.diffusion.params.timestep_respacing = args.timestep_respacing

    # build the sampler for diffusion
    configs.model_ir.ckpt_path = args.ckpt_path
    sampler_dist = DifIRSampler(configs)

    # prepare low quality images
    exts_all = ('jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'bmp')
    if args.in_path.endswith(exts_all):
        im_path_list = [Path(args.in_path), ]
    else: # for folder
        im_path_list = []
        for ext in exts_all:
            im_path_list.extend([x for x in Path(args.in_path).glob(f'*.{ext}')])

    im_path_list = sorted(im_path_list)
    
    # prepare gt images
    if args.gt_path is not None:
        if args.gt_path.endswith(exts_all):
            gt_path_list = [Path(args.in_path), ]
        else: # for folder
            gt_path_list = []
            for ext in exts_all:
                gt_path_list.extend([x for x in Path(args.gt_path).glob(f'*.{ext}')])
        gt_path_list = sorted(gt_path_list)
        
        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        psnr_list, ssim_list, lpips_list = [], [], []
        filename_list = []
        run_eval = True
    else:
        run_eval = False

    # prepare result path
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)
    restored_face_dir = Path(args.out_path) / 'restored_faces'
    if not restored_face_dir.exists():
        restored_face_dir.mkdir()
    if not args.aligned:
        cropped_face_dir = Path(args.out_path) / 'cropped_faces'
        if not cropped_face_dir.exists():
            cropped_face_dir.mkdir()
        restored_image_dir = Path(args.out_path) / 'restored_image'
        if not restored_image_dir.exists():
            restored_image_dir.mkdir()

    for ii, im_path in enumerate(im_path_list):
        if (ii+1) % 5 == 0:
            print(f"Processing: {ii+1}/{len(im_path_list)}...")
        im_lq = util_image.imread(im_path, chn='bgr', dtype='uint8')
        if run_eval:
            im_gt = util_image.imread(gt_path_list[ii], chn='bgr', dtype='uint8')
            im_gt = img2tensor(im_gt, bgr2rgb=True, float32=True).unsqueeze(0) / 255.
            im_gt = im_gt.cuda()
        if args.aligned:
            face_restored = sampler_dist.restore_func_ir_aligned(
                    y0=im_lq,
                    ) #[0,1], 'rgb'
            if run_eval:
                psnr_curr = util_image.calculate_psnr_single_img(face_restored, im_gt, process=True)
                ssim_curr = util_image.calculate_ssim_single_img(face_restored, im_gt, process=True)
                lpips_curr = lpips_fn(face_restored, im_gt, normalize=True).item()
            face_restored = util_image.tensor2img(
                    face_restored,
                    rgb2bgr=True,
                    min_max=(0.0, 1.0),
                    ) # uint8, BGR
            save_path = restored_face_dir / im_path.name
            util_image.imwrite(face_restored, save_path, chn='bgr', dtype_in='uint8')

        
        if run_eval:
            filename_list.append(im_path.stem)
            psnr_list.append(psnr_curr)
            ssim_list.append(ssim_curr)
            lpips_list.append(lpips_curr)
    
    if run_eval:
        psnr_mean = np.mean(np.array(psnr_list))
        ssim_mean = np.mean(np.array(ssim_list))
        lpips_mean = np.mean(np.array(lpips_list))
        
        output_text_file = Path(args.out_path) / 'output.txt'
        with open(output_text_file, 'a') as f:
            f.write('Evaluation\n')
            f.write(f'Average PSNR {psnr_mean}\n')
            f.write(f'Average SSIM {ssim_mean}\n')
            f.write(f'Average LPIPS {lpips_mean}\n')
            f.write('filename | PSNR | SSIM | LPIPS\n')
            for file_idx in range(len(filename_list)):
                f.write(f'{filename_list[file_idx]} | {psnr_list[file_idx]} | {ssim_list[file_idx]} | {lpips_list[file_idx]}\n')
        
        output_pickle_file = Path(args.out_path) / 'output.pkl'
        with open(output_pickle_file, 'wb') as pkl_file:
            pickle.dump({'filenames': filename_list, 
                         'psnr': np.array(psnr_list),
                         'ssim': np.array(ssim_list),
                         'lpips': np.array(lpips_list)}, pkl_file)

if __name__ == '__main__':
    main()