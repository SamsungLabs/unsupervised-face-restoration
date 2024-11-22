import glob
from argparse import Namespace
import random
from pathlib import Path

import os
import torch
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from argparse import ArgumentParser
import cv2
import numpy as np
from demo_ccm_sampling import main as graphic2raw
from noise_profiler.image_synthesizer import load_noise_model
import pickle
import demosaicnet
from utils.raw_utils import *
from utils.image_utils import apply_ccm, apply_wbgain, gamma
from utils.image_utils import demosaic as demosaic_opencv
from utils.imresize_kernelgan import imresize as imresize_with_kernel
from scipy.io import loadmat

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def render(img_demosaic_rgb, wb_gain, ccm):
    # Apply WB and CCM
    img_rgb_wb_ccm = apply_wbgain(img_demosaic_rgb, wb_gain).clip(0, 1)
    img_rgb_wb_ccm = apply_ccm(img_rgb_wb_ccm, ccm).clip(0, 1)
    img_rgb_wb_ccm = img_rgb_wb_ccm ** (1 / 2.2)  # gamma correction
    return img_rgb_wb_ccm


def black_level_subtraction(img_raw):
    return np.clip((img_raw - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL), 0, 1)  # [0, 1]


def add_noise_to_raw(img_raw_clean, noise_model, iso, is_tetra):
    if is_tetra:
        img_raw_noisy = add_noise_to_tetra_raw(img_raw_clean, noise_model, iso)
    else:
        img_raw_noisy = add_noise_to_bayer_raw(img_raw_clean, noise_model, iso)
    return img_raw_noisy


def process_raw_for_visualization(img_raw_noisy, is_tetra, cfa_pattern):
    if is_tetra:
        img_raw_noisy_bgr = process_tetra_raw_for_visualization(img_raw_noisy, cfa_pattern)
    else:
        img_raw_noisy_bgr = process_bayer_raw_for_visualization(img_raw_noisy, cfa_pattern)
    return img_raw_noisy_bgr


def demosaic(img_raw_bgr, type_demosaic='demosaicnet', cfa_pattern='GRBG'):
    """
    Demosaic noisy RAW image.
    The demosaicing network has been trained with GRBG cfa patterns. When generating data for a sensor with a GBRG
    cfa pattern, we need to flip the input before passing to the demosaicnet, to go from GBRG --> GRBG and then
    flip the output again to go back to GRBG. We also need to invert the channel order of the input before passing it
    to the network, because it assume RGB inputs.
    :param img_raw_bgr:  Input noisy RAW in !!! BGR !!! format.
    :param type_demosaic: 'opencv' or 'demosaicnet' (default)
    :param cfa_pattern: 'GRBG' (== '1') or 'GBRG' (== '2')
    :return: Demosaiced output in RGB format.
    """

    if type_demosaic == 'opencv':
        img_demosaic_rgb = demosaic_opencv(img_raw_bgr, cfa_pattern, bit_size=8)
    elif type_demosaic == 'demosaicnet':
        if cfa_pattern in ['GBRG', '2']:
            img_raw_bgr = np.transpose(img_raw_bgr, (1, 0, 2))
        with torch.no_grad():
            raw_in = torch.from_numpy(np.transpose(img_raw_bgr[:, :, ::-1], [2, 0, 1]).astype(np.float32)).unsqueeze(0)
            img_demosaic_rgb = demosaic_net(raw_in.to(device)).squeeze(0).cpu().numpy().transpose((1, 2, 0)).clip(0, 1)
        if cfa_pattern in ['GBRG', '2']:
            img_demosaic_rgb = np.transpose(img_demosaic_rgb, (1, 0, 2))
    else:
        raise ValueError('Invalid demosaic type.')

    return img_demosaic_rgb


def BGR_HR_to_BGR_LR(img_bgr, scale, kernel_directory=None):

    def get_downsampling_kernel(kernel_directory):
        # Randomly pick one of the available kernels.
        kernel_list = glob.glob(os.path.join(kernel_directory, f'**/*x{scale}.mat'), recursive=True)
        kernel_file = random.choice(kernel_list)
        print(f'Downsampling using kernel {kernel_file}...')
        kernel = loadmat(kernel_file)['Kernel']
        return kernel

    # downsample RGB image by a factor of "args.scale" before converting to tetra RAW
    new_size = (img_bgr.shape[1] // scale, img_bgr.shape[0] // scale)
    if kernel_directory is None:
        img_bgr_downsampled = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA).clip(0, 1)
    else:
        kernel = get_downsampling_kernel(kernel_directory)
        img_bgr_downsampled = imresize_with_kernel(img_bgr, 1.0/scale, output_shape=new_size[::-1], kernel=kernel)
    return img_bgr_downsampled


def BGR_LR_to_RAW_LR(img_bgr_downsampled, args):
    if args.tetra_raw:
        raw_LR = BGR2TetraRAW(img_bgr_downsampled, args.cfa_pattern) * WHITE_LEVEL
    else:
        # raw_LR = BGR2BayerRAW(img_bgr_downsampled, args.cfa_pattern) * WHITE_LEVEL
        raw_LR = BGR2BayerRAW(img_bgr_downsampled, args.cfa_pattern)
    # raw_LR = np.clip(change_blacklevel(raw_LR, new_bl=BLACK_LEVEL), 0, WHITE_LEVEL)
    raw_LR = raw_LR * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL
    return raw_LR


def bgr_to_linear_bgr(exr_img, args):
    # The EXR data are in a linearized RGB color space. For all other publicly available RGB data, we de-gamma.
    if args.dataset != 'exr':
        exr_img = exr_img ** 2.2  # de-gamma

    # Invert CCM and WB to bring graphics image to raw space
    sampled_ccm, sampled_illum, raw_img_no_wb = graphic2raw(args.dictionary_path, args.sampling_mode, args.k, exr_img=exr_img)
    sampled_wbgain = 1. / sampled_illum
    sampled_wbgain = (sampled_wbgain[0], sampled_wbgain[1], sampled_wbgain[1], sampled_wbgain[2])
    return raw_img_no_wb, sampled_ccm, sampled_wbgain


def load_noise_model_for_sensor(sensor, ISO, is_tetra):
    def get_hg_noise_model(noise_model_path):
        noise_model, iso2b1_interp_splines, iso2b2_interp_splines = load_noise_model(path=noise_model_path)
        noise_model = {'noise_model': noise_model,
                       'iso2b1_interp_splines': iso2b1_interp_splines,
                       'iso2b2_interp_splines': iso2b2_interp_splines}
        return noise_model

    def get_ITS_noise_model_for_iso_GN3_sensor(iso, is_tetra):
        if is_tetra:
            noise_models_dir = 'non_parametric_noise_modeling_s22p_GN3_Tetra_ISOs100-2000'
            noise_model_iso_dir = 'saved_inverse_transforms_calibration_stack_clean_iso50_s22+tetrabinary_ISO_'
        else:
            noise_models_dir = 'non_parametric_noise_modeling_s22p_GN3_Full_ISOs50-6400'
            noise_model_iso_dir = 'saved_inverse_transforms_calibration_stack_clean_iso20_s22+GN3_12MP_ISO_'
        isos = np.array([int(d.split('ISO_')[-1]) for d in os.listdir(noise_models_dir) if 'ISO_' in d])
        idx_min = np.argmin(np.abs(isos - iso))
        iso_closest = isos[idx_min]
        noise_model_path = os.path.join(noise_models_dir, noise_model_iso_dir + f'{iso_closest}',
                                        f'inverse_transform_iso_{iso_closest}_max_bins_False_num_bins_128_incr_4.p')
        noise_model = pickle.load(open(noise_model_path, "rb"))  # read noise model
        return noise_model

    if sensor.upper() == 'IMX754':
        # For the IMX754 sensor we use a heteroscedastic gaussian (HG) noise model (not available for tetra RAW).
        # HG has been calibrated for a wide range of ISOs so we don't need to load different models per ISO value.
        if is_tetra:
            raise NotImplementedError
        noise_model = get_hg_noise_model('./noise_profiler/hg-s22u-s908u-na-b0q-pr-t2-id35248-v7')
    elif sensor.upper() in ['GN3', 'S5KGN3']:
        # For the GN3 sensor we can use either HG or ITS model (available for both Bayer and Tetra RAW).
        # ITS models are calibrated for a specific ISO value. In that case, we load the model corresponding to the ISO
        # that is closest to the desired ISO value.
        noise_model = get_hg_noise_model('./noise_profiler/hg-s22p-s906u-na-g0q-pr-w-id35249-v7')  # HG model
        # noise_model = get_ITS_noise_model_for_iso_GN3_sensor(ISO, is_tetra)
    else:
        raise NotImplementedError
    return noise_model


def sensor_to_cfa_pattern(sensor):
    if sensor.upper() == 'IMX754':
        return 'GRBG'  # (type 1)
    elif sensor.upper() in ['GN3', 'S5KGN3']:
        return 'GBRG'  # (type 2)


def get_compact_exr_name(exr_fn):
    """
    :param exr_dir:
    :param exr_fn:
    Example
    exr_dir/ModernLiving_DebugBatch1_mfp_iso50_aligned/ GT_26_50/GT_26_4200x3200_output_GT_RGB.exr'
    :return: ModernLiving_26
    """
    exr_subdirs = exr_fn.split(os.path.sep)
    exr_name = exr_subdirs[-1]
    exr_index = exr_name.split('_')[1]
    scene_name = exr_subdirs[-3]
    scene_name = scene_name[:scene_name.index('_DebugBatch1')]
    compact_name = f'{scene_name}_{exr_index}'
    return compact_name


def read_and_pad_image(scene_fn, args):
    """
    Read image and make sure image size is divisible by the downsampling scale * cfa_factor
    E.g., for a tetra pattern and a downsampling rate of 4, we want the frames to be divisible by 4*4=16,
    to be able to downscale and extract the Tetra RAW pattern.
    For a Bayer pattern, and a downsampling rate of 4, we want the frames to be divisible by 2*4=8.
    However, for simplicity, reusability of the HR data, we always use a cfa_factor of 4, which works for both cases.
    We pad with zeros (we also assume that all frames in the burst have the same dimensions).
    :param scene_fn:
    :param args:
    :return: padded image BGR format
    """
    if args.dataset == 'exr':
        img_bgr = np.clip(cv2.imread(scene_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), 0, 1)  # [0, 1]
    else:
        img_bgr = np.clip(cv2.imread(scene_fn, cv2.IMREAD_COLOR) / 255, 0, 1)  # [0, 1]
    return img_bgr


def create_directories(args):
    # Create all directories needed for storing intermediate and final results
    cfa_pattern = 'tetra' if args.tetra_raw else 'bayer'
    raw_dir = Path(args.results_dir) / f'{args.dataset}_scale_x{args.scale}_{args.sensor}_{cfa_pattern}_ISO{args.iso_range[0]}-{args.iso_range[1]}'
    dir_rendered = raw_dir / f'demosaic_{cfa_pattern}_rendered'

    raw_dir.mkdir(parents=True, exist_ok=True)
    dir_rendered.mkdir(parents=True, exist_ok=True)

    return dir_rendered


if __name__ == '__main__':
    parser = ArgumentParser()

    # Sensor and input directory options.
    parser.add_argument('--exr_dir', type=str, default=str(Path.home() / 'datasets' / 'graphics-dataset-sidi_small'),
                        help='Root path to exr images to be processed. Assume a specific directory structure.')
    parser.add_argument('--hq_dir', type=str, default=str(Path.home() / 'datasets' / 'DIV2K' / 'DIV2K_train_HR'),
                        help='Root path to HQ images to be processed. Assumes a specific directory structure.')
    parser.add_argument('--flickr2k_dir', type=str, default=str(Path.home() / 'datasets' / 'Flickr2K'),
                        help='Root path to Flickr2K HR images to be processed. Assumes a specific directory structure.')
    parser.add_argument('--results_dir', type=str, default=str(Path.home() / 'datasets'),
                        help='Path where generated data are stored')
    parser.add_argument('--kernel_dir', default=None,
                        help='Directory where downsampling kernels (extracted with KernelGAN) are stored.')
    parser.add_argument('--sensor', type=str, default='IMX754', choices=['IMX754', 'GN3'],
                        help='Sensor used. Determines the bayer pattern and noise model used.')

    # Options related to graphics2raw inversion pipeline.
    parser.add_argument('--txt_template', type=str, default='./metadata_templates',
                        help='Directory containing configuration template txt file')
    parser.add_argument('--dictionary_path', type=str, default='./data/ccm_awb_dictionary_52_plus_230.p',
                        help='Path to CCM and WB dictionary that holds the pool of CCMs and illumination vectors to sample from.')
    parser.add_argument('--sampling_mode', type=str, default='knn',
                        help='Type of ccm sampling. Options: discrete or knn.')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of nearest neighbours, applies only for knn sampling_mode.')

    # Noise, super resolution, and rendering options.
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--iso_min', type=int, default=0, help='Min value of ISO used to generate synthetic noise.')
    parser.add_argument('--iso_max', type=int, default=1000, help='Max value of ISO used to generate synthetic noise.')
    parser.add_argument('--tetra_raw', action='store_true', help='Generate synthetic data for Tetra RAW.')
    parser.add_argument('--gamma', type=str, choices={'default', 'davinci', '2.4'}, default='default')
    parser.add_argument('--dataset', type=str, choices={'exr', 'div2k', 'flickr2k'}, default='div2k')

    args = parser.parse_args()
    args.cfa_pattern = sensor_to_cfa_pattern(args.sensor)

    # Define ISO range. During data generation, ISO is randomly sampled within this range to generate synthetic noise.
    args.iso_range = [args.iso_min, args.iso_max]

    # Define demosaicing networks
    demosaic_net = demosaicnet.BayerDemosaick(pad=1, tetra=args.tetra_raw).to(device)

    # Read all RGB HR image filenames
    if args.dataset == 'exr':
        all_scenes = sorted(glob.glob(os.path.join(args.exr_dir, '**', '*.exr'), recursive=True))
    elif args.dataset == 'div2k':
        all_scenes = sorted(glob.glob(os.path.join(args.hq_dir, '**', '*.png'), recursive=True))
    elif args.dataset == 'flickr2k':
        all_scenes = sorted(glob.glob(os.path.join(args.flickr2k_dir, '**', '*.png'), recursive=True))
    print(f'Found {len(all_scenes)} scenes from {args.dataset} dataset.')
    print(f'Generating synthetic data for sensor {args.sensor}, {"Tetra" if args.tetra_raw else "Bayer"} RAW.')
    print(f'ISO for synthetic noise in the [{args.iso_min}, {args.iso_max}] range.')

    # Create all directories for storing generated data.
    dir_rendered = create_directories(args)

    for j, scene_fn in enumerate(all_scenes, 1):

        # Sample ISO and load appropriate noise model.
        ISO = random.randint(args.iso_range[0], args.iso_range[1])  # randomly sample ISO value in range
        noise_model = load_noise_model_for_sensor(args.sensor, ISO, is_tetra=args.tetra_raw)

        # Load clean RGB image and pad it to make sure its dimensions are compatible with the downscaling factor.
        raw_name = get_compact_exr_name(scene_fn) if args.dataset == 'exr' else os.path.splitext(os.path.basename(scene_fn))[0]
        print(f'----Processing {raw_name}---- ({j}/{len(all_scenes)})')

        if (dir_rendered / (raw_name + '.png')).exists():
            continue

        img_bgr = read_and_pad_image(scene_fn, args)  # [0, 1]

        img_bgr_linear, sampled_ccm, sampled_wbgain = bgr_to_linear_bgr(img_bgr, args)

        img_bgr_LR = BGR_HR_to_BGR_LR(img_bgr_linear, args.scale, kernel_directory=args.kernel_dir)

        img_raw_clean = BGR_LR_to_RAW_LR(img_bgr_LR, args)  # [BLACK_LEVEL, WHITE_LEVEL]

        img_raw_noisy = add_noise_to_raw(img_raw_clean, noise_model, ISO, args.tetra_raw)  # [0, WHITE_LEVEL]

        # Convert RAW to 3-channel BGR RAW images (used as input to demosaic net)
        img_raw_noisy_bgr = process_raw_for_visualization(img_raw_noisy, args.tetra_raw, args.cfa_pattern)  # [0, WHITE_LEVEL]

        img_raw_noisy_bgr = black_level_subtraction(img_raw_noisy_bgr)  # [0, 1]

        img_demosaic_rgb = demosaic(img_raw_noisy_bgr, 'demosaicnet', cfa_pattern=args.cfa_pattern)

        img_rendered_rgb = render(img_demosaic_rgb, sampled_wbgain, sampled_ccm)
        
        img_rendered_rgb = cv2.resize(img_rendered_rgb, (512, 512)).clip(0, 1)

        # Save all results. Since we are using opencv to write images, we need to convert RGB -> BGR before saving.
        fn_png = raw_name + '.png'
        cv2.imwrite(str(dir_rendered / fn_png), np.uint8(img_rendered_rgb[:, :, ::-1] * (2 ** 8 - 1)))

