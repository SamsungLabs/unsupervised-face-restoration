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
import shutil

from utils import util_opts
from utils import util_image
from utils import util_common

from sampler import DifIRSampler
from ResizeRight.resize_right import resize
from basicsr.utils.download_util import load_file_from_url
from resizer import Resizer
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
            "--apply_filter",
            action='store_true',
            help='Apply low pass filter during denoising process',
            )
    parser.add_argument(
            "--save_restoration_output",
            action='store_true',
            help='save restoration',
            )
    parser.add_argument(
            "--end_timesteps",
            type=int,
            default=None,
            help='end timestep for DifFace before performing one-step denoising, parameter N in our paper (Default:0)',
            )
    parser.add_argument(
            "--reg_end_timesteps",
            type=int,
            default=0,
            help='end timestep for DifFace with low freq content regularization, parameter N in our paper (Default:0)',
            )
    parser.add_argument(
            "--down_N",
            type=int,
            default=None,
            help='downsampling and upsampling factor for linear filter',
            )
    parser.add_argument(
            "--guidance_scale",
            type=float,
            default=None,
            help='guidance scale for guiding the denoising process using linear filter',
            )
    parser.add_argument(
            "--gt_path",
            type=str,
            default=None,
            help='Folder to GT images',
            )
    parser.add_argument(
            "--config_path",
            type=str,
            default=None,
            help='config file for diffusion model and restoration model',
            )
    args = parser.parse_args()

    if args.config_path is None:  
        cfg_path = 'configs/sample/iddpm_ffhq512_swinir.yaml'
    else:
        cfg_path = args.config_path

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = args.gpu_id
    configs.aligned = args.aligned
    assert args.started_timesteps < int(args.timestep_respacing)
    configs.diffusion.params.timestep_respacing = args.timestep_respacing

    # prepare the checkpoint
    if not Path(configs.model.ckpt_path).exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth",
            model_dir=str(Path(configs.model.ckpt_path).parent),
            progress=True,
            file_name=Path(configs.model.ckpt_path).name,
            )
    if not Path(configs.model_ir.ckpt_path).exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
            model_dir=str(Path(configs.model_ir.ckpt_path).parent),
            progress=True,
            file_name=Path(configs.model_ir.ckpt_path).name,
            )

    # build the sampler for diffusion
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

    if args.save_restoration_output:
        restoration_output_dir = Path(args.out_path) / 'ir'
        restoration_output_dir.mkdir()

    if args.apply_filter:
        down_N = args.down_N
        img_shape = 512
        shape = (1, 3, img_shape, img_shape)
        shape_d = (1, 3, img_shape // down_N, img_shape // down_N)
        down = Resizer(shape, 1 / down_N).cuda()
        up = Resizer(shape_d, down_N).cuda()
        filter_dict = {'down':down, 'up':up}
        if args.guidance_scale is not None:
            filter_dict['guidance_scale'] = args.guidance_scale
    else:
        filter_dict = None

    for ii, im_path in enumerate(im_path_list):
        if (ii+1) % 5 == 0:
            print(f"Processing: {ii+1}/{len(im_path_list)}...")
        im_lq = util_image.imread(im_path, chn='bgr', dtype='uint8')
        # shutil.copy(str(im_path), (restored_face_dir / ('input_' + im_path.name)))
        if run_eval:
            im_gt = util_image.imread(gt_path_list[ii], chn='bgr', dtype='uint8')
            im_gt = img2tensor(im_gt, bgr2rgb=True, float32=True).unsqueeze(0) / 255.
            im_gt = im_gt.cuda()
        if args.aligned:
            face_restored, ir_output = sampler_dist.sample_func_ir_aligned(
                    y0=im_lq,
                    start_timesteps=args.started_timesteps,
                    need_restoration=True,
                    filter_dict=filter_dict,
                    end_timesteps=args.end_timesteps,
                    reg_end_timesteps=args.reg_end_timesteps,
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

            if args.save_restoration_output:
                # save restoration model output
                ir_output = util_image.tensor2img(
                        ir_output,
                        rgb2bgr=True,
                        min_max=(0.0, 1.0),
                        ) # uint8, BGR
                ir_save_path = restoration_output_dir / im_path.name
                util_image.imwrite(ir_output, ir_save_path, chn='bgr', dtype_in='uint8')

        else:
            image_restored, face_restored, face_cropped = sampler_dist.sample_func_bfr_unaligned(
                    y0=im_lq,
                    start_timesteps=args.started_timesteps,
                    need_restoration=True,
                    draw_box=args.draw_box,
                    )

            # save the whole image
            save_path = restored_image_dir / im_path.name
            util_image.imwrite(image_restored, save_path, chn='bgr', dtype_in='uint8')

            # save the cropped and restored faces
            assert len(face_cropped) == len(face_restored)
            for jj, face_cropped_current in enumerate(face_cropped):
                face_restored_current = face_restored[jj]

                save_path = cropped_face_dir / f"{im_path.stem}_{jj}.png"
                util_image.imwrite(face_cropped_current, save_path, chn='bgr', dtype_in='uint8')

                save_path = restored_face_dir / f"{im_path.stem}_{jj}.png"
                util_image.imwrite(face_restored_current, save_path, chn='bgr', dtype_in='uint8')

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
