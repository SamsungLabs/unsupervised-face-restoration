import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils.metadata_utils import read_mfp_input_metadata, read_mfp_input_raw_image


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def render_pngraw(img_raw, demosaic='opencv', gamma_type=None, bayer_pattern=None, after_mfp=True):
    """
    Takes a RAW file, saved as png and renders it
    :param img_raw:
    :return:
    """
    # Normalize, clip.
    c_norm = 4095.0 if after_mfp else 1023.0  # input to MFP is 10-bit but output is 12-bit
    img_raw = img_raw / c_norm
    img_raw = np.clip(img_raw, 0.0, 1.0)

    # Demosaic (stacked RGGB to visualization)
    if demosaic == 'opencv':
        max_val = 16383
        opencv_demosaic_flag = get_opencv_demosaic_flag(cfa_pattern=bayer_pattern,
                                                        output_channel_order='RGB', alg_type='EA')
        img_demosaic = (img_raw * max_val).astype(dtype=np.uint16)
        img_demosaic = cv2.cvtColor(img_demosaic, opencv_demosaic_flag)
        img_demosaic = img_demosaic.astype(dtype=np.float32) / max_val
    else:
        raise NotImplementedError

    if gamma is not None:
        img_demosaic = gamma(img_demosaic, gamma_type)

    return img_demosaic


def get_opencv_demosaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    """
    using opencv edge-aware demosaicing
    Flag correnspondence documentation: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    """
    assert output_channel_order.upper() in ['RGB', 'BGR']
    if alg_type != '':
        alg_type = '_' + alg_type
    if cfa_pattern == [0, 1, 1, 2] or cfa_pattern == 'RGGB':
        opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2' + output_channel_order.upper() + alg_type)
    elif cfa_pattern == [2, 1, 1, 0] or cfa_pattern == 'BGGR':
        opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2' + output_channel_order.upper() + alg_type)
    elif cfa_pattern == [1, 0, 2, 1] or cfa_pattern == 'GRBG':
        opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2' + output_channel_order.upper() + alg_type)
    elif cfa_pattern == [1, 2, 0, 1] or cfa_pattern == 'GBRG':
        opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2' + output_channel_order.upper() + alg_type)
    else:
        opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2' + output_channel_order.upper() + alg_type)
        print("CFA pattern not identified.")
    return opencv_demosaic_flag


def raw_file_to_png_file(raw_path, png_path, output_shape):
    """
    Save a file ending with .raw as .png, no other changes
    :param raw_path:
    :return:
    """
    raw = np.fromfile(raw_path, dtype="uint16")
    raw = np.reshape(raw, output_shape)
    cv2.imwrite(png_path, raw)


def stack_bayer(bayer):
    h, w = bayer.shape
    stack = np.empty((int(h / 2), int(w / 2), 4), dtype=np.float32)

    stack[:, :, 0] = bayer[0::2, 0::2]  # top left
    stack[:, :, 1] = bayer[0::2, 1::2]  # top right
    stack[:, :, 2] = bayer[1::2, 0::2]  # bottom left
    stack[:, :, 3] = bayer[1::2, 1::2]  # bottom right

    return stack


def invert_stack_bayer(stack):
    h, w = stack.shape[:-1]
    bayer = np.empty((int(h * 2), int(w * 2)), dtype=np.float32)

    bayer[0::2, 0::2] = stack[:, :, 0]  # top left
    bayer[0::2, 1::2] = stack[:, :, 1]  # top right
    bayer[1::2, 0::2] = stack[:, :, 2]  # bottom left
    bayer[1::2, 1::2] = stack[:, :, 3]  # bottom right

    return bayer


def tone_map(image, black_level, white_level, alg='drago'):
    """
    :param image:
    :param black_level:
    :param white_level:
    :param alg:
    :return: Quantized LDR image in the range of [black level, white level], 16 bits
    """
    if alg == 'drago':
        image = image.astype(np.float32)
        drago = cv2.createTonemapDrago(1)
        image = drago.process(image) * 0.3
        image = image * (white_level - black_level) + black_level
    elif alg == 'he':
        image = hist_eq(image, black_level, white_level)
    else:
        print('No tone mapping.')
        image = image * (white_level - black_level) + black_level

    image = image.astype(np.uint16)
    return image


def hist_eq(image, black_level, white_level):
    """
    Source: https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43
    """
    image = (image * (2 ** 32)).astype(np.uint32)  # quantize image to uint 32

    # STEP 1: Normalized histogram
    histogram_array = np.bincount(image.flatten(), minlength=white_level)  # histogram values
    # normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels  # get a probability distribution

    # cumulative histogram
    cdf_array = np.cumsum(histogram_array)

    # STEP 2: Pixel mapping lookup table
    # maps uint 32 values to [black_level, white_level] values
    cdf_array = cdf_array * (white_level - black_level) + black_level
    transform_map = np.floor(cdf_array).astype(np.uint16)

    # STEP 3: transformation
    equalized_image = transform_map[image.flatten()]
    # reshape and write back into img_array
    equalized_image = np.reshape(np.asarray(equalized_image), image.shape)
    return equalized_image


def count_bit_depth(image, channel=0):
    bit_depth = np.log2(len(np.unique(image[..., channel])))
    return bit_depth


def gamma(image, type_gamma='default'):
    if type_gamma is None:
        pass
    elif type_gamma == 'default':
        image = image ** (1 / 2.2)
    elif type_gamma == 'davinci':
        gamma_file = 'data/One_DavinciGamma_v1.txt'
        image = gamma_davinci(image, gamma_file)
    elif type_gamma == '2.4':
        image = gamma_24(image)
    else:
        raise Exception('Unrecognized gamma: ', type_gamma)
    return image


def gamma_davinci(image, gamma_file):
    f = open(gamma_file, "r")
    lines = f.readlines()
    f.close()

    max_val = 2 ** 14 - 1
    lines = [l.strip().split(',') for l in lines]
    x = np.array([int(l[0]) for l in lines])
    y = np.array([int(l[1]) for l in lines])

    xq = np.linspace(0, max_val, num=max_val + 1)
    gamma_lut = interp1d(x, y, kind='cubic')(xq)

    image = np.round(image * max_val).astype(np.uint16)
    image = gamma_lut[image]
    image = image.astype(np.float32) / max_val
    return image


def gamma_24(image):
    """
    Source: /Users/lucy.zhao/Projects/sidi_exr_to_raw_ccm_donghwan_april3_2023,
    function: linear2sRGB
    Converts a linear sRGB file into sRGB.
    """
    return np.where(
        image <= 0.0031308, 12.92 * image, ((1 + 0.055) * image ** (1 / 2.4)) - 0.055
    )


def demosaic(image, cfa_pattern):
    max_val = 16383
    image = (image * max_val).astype(dtype=np.uint16)
    opencv_demosaic_flag = get_opencv_demosaic_flag(cfa_pattern, output_channel_order='RGB', alg_type='EA')
    image = cv2.cvtColor(image, opencv_demosaic_flag)
    image = image.astype(dtype=np.float32) / max_val
    return image


def white_balance_rgb(image_rgb, illuminant, clip=True):
    illuminant_ = illuminant.copy()
    illuminant_ = np.reshape(illuminant_, (1, 1, 3))
    white_balanced_image = image_rgb / illuminant_
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def apply_wbgain(img, wb_gain):
    """
    Applies white balance (WB) gain factors (assumes WB is stored in RGB format)
    :param img: HxWx3 image (RGB format)
    :param wb_gain: 1x4 vector (RGGB)
    :return:
    """
    wb_gain = np.array([wb_gain[0], wb_gain[1], wb_gain[3]])
    return img * wb_gain.reshape((1, 1, 3))


def apply_inverse_ccm_on_image(image, forward_ccm):
    forward_ccm = np.reshape(np.asarray(forward_ccm), (3, 3))
    forward_ccm = forward_ccm / np.sum(forward_ccm, axis=1, keepdims=True)
    inverse_ccm = np.linalg.inv(forward_ccm)
    image = inverse_ccm[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
    image = np.sum(image, axis=-1)
    return image


def apply_ccm_on_image(image, forward_ccm):
    forward_ccm = np.reshape(np.asarray(forward_ccm), (3, 3))
    # forward_ccm = forward_ccm / np.sum(forward_ccm, axis=1, keepdims=True)
    image = forward_ccm[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
    image = np.sum(image, axis=-1)
    return image


def apply_ccm(image, ccm):
    """ Applies a color correction matrix. (RGB format)
    :param image: HxWx3 numpy.array
    :param ccm:  3x3 numpy.array
    :return: HxWx3 numpy.array
    """
    return (image.reshape(-1, 3) @ ccm.transpose()).reshape(image.shape)


def save_image(image, save_path, bit_depth=8):
    image = (image * (2**bit_depth-1)).astype(f'uint{bit_depth}')
    cv2.imwrite(save_path, image[:, :, ::-1])


def visualize_mfp_output_raw(image_fn, cfa_pattern, save_fn=None, do_save=True, is_png=False, gamma_type='default', bit_depth=8):
    black_level = 0  # no black level subtraction
    white_level = 4095
    if is_png:
        image = np.array(cv2.imread(image_fn, -1)).astype(np.float32) / 65535.0
        # Already normalized
    else:
        image = read_mfp_input_raw_image(image_fn)
        # Normalization
        image = (image - black_level) / (white_level - black_level)
        image = np.clip(image, 0, 1)

    # Demosaic
    image = demosaic(image, cfa_pattern)

    # Gamma
    image = gamma(image, type_gamma=gamma_type)

    if do_save:
        save_path = save_fn if save_fn else image_fn[:-4] + '_ispsim.png'
        save_image(image, save_path, bit_depth)
    return image


def visualize_mfp_input_raw(image_fn, cfa_pattern, save_fn=None, meta_fn=None, do_save=True, gamma_type='default', raw_image=None, bit_depth=8):
    """
    :param image_fn:
    :param cfa_pattern:
    :param save_fn:
    :param meta_fn: if not None, perform white balance
    :param do_save: whether to save the processed image or return it
    :return:
    """
    black_level = 64
    white_level = 1023
    image = read_mfp_input_raw_image(image_fn) if raw_image is None else raw_image.astype(np.float32)

    print(f'Image max: {image.max()}, image min: {image.min()}')

    # Normalization
    image = (image - black_level) / (white_level - black_level)
    image = np.clip(image, 0, 1)

    # Demosaic
    image = demosaic(image, cfa_pattern)

    # White balance (optional)
    if meta_fn:
        metadata = read_mfp_input_metadata(meta_fn)
        assert metadata['i32Width'] == image.shape[1] and metadata['i32Height'] == image.shape[0]
        wb_vec = metadata['as_shot_neutral']
        image = white_balance_rgb(image, wb_vec, clip=True)

    # Gamma
    image = gamma(image, type_gamma=gamma_type)

    if do_save:
        save_path = save_fn if save_fn else image_fn[:-4] + '_ispsim.png'
        save_image(image, save_path, bit_depth=bit_depth)
    return image


def get_cfa_pattern(cfa_pattern_str):
    rgb_dict = {'r': 0, 'g': 1, 'b': 2}
    cfa_pattern = [rgb_dict[c] for c in cfa_pattern_str.lower()]
    return cfa_pattern


def visualize_mfp_raw(args):
    def try_find_input_meta(in_file):
        # try to find the input metadata file if not provided
        input_meta_fn = in_file[:-len('4080x3060.raw')] + 'Args.txt'
        if os.path.exists(input_meta_fn):
            print('Using ', input_meta_fn)
            return input_meta_fn
        else:
            return ''

    cfa_pattern = get_cfa_pattern(args.cfa_pattern)
    if args.type == 'mfp_output':
        if os.path.isfile(args.input):
            is_png = args.input.endswith('.png')
            visualize_mfp_output_raw(args.input, cfa_pattern, args.output, is_png=is_png, gamma_type=args.gamma_type)
        else:
            input_files = glob.glob(os.path.join(args.input, '*.raw'))
            is_png = len(input_files) == 0
            if is_png:
                input_files = glob.glob(os.path.join(args.input, '*.png'))
            for input_file in input_files:
                visualize_mfp_output_raw(input_file, cfa_pattern, args.output, is_png=is_png, gamma_type=args.gamma_type)
    elif args.type == 'mfp_input':
        if os.path.isfile(args.input):
            input_meta_fn = try_find_input_meta(args.input) if not args.input_meta_fn else args.input_meta_fn
            visualize_mfp_input_raw(args.input, cfa_pattern, args.output, input_meta_fn, gamma_type=args.gamma_type)
        else:
            input_files = sorted(glob.glob(os.path.join(args.input, '*.raw')))
            if args.sample_inputs:
                input_files = input_files[:4]  # pick EV-4, EV-2, and 1 EV0 frame to visualize
            for input_file in input_files:
                input_meta_fn = try_find_input_meta(input_file)
                visualize_mfp_input_raw(input_file, cfa_pattern, args.output, input_meta_fn,
                                        gamma_type=args.gamma_type)
    elif args.type == 'mfp_input_dir':
        """
        top_level_dir
            mfp_input_burst1
            mfp_input_burst2
            ...
        """
        input_dirs = get_immediate_subdirectories(args.input)
        for input_dir in input_dirs:
            print('Input dir ', input_dir)
            input_files = sorted(glob.glob(os.path.join(args.input, input_dir, '*.raw')))
            save_fn = None
            if args.sample_one:
                input_files = [input_files[-1]]  # pick 1 EV0 frame to visualize
                save_fn = os.path.join(args.input, f'{input_dir}_ispsim.png')
            if args.sample_inputs:
                input_files = input_files[:4]  # pick EV-4, EV-2, and 1 EV0 frame to visualize
            for input_file in input_files:
                input_meta_fn = try_find_input_meta(input_file)
                visualize_mfp_input_raw(input_file, cfa_pattern, save_fn, input_meta_fn,
                                        gamma_type=args.gamma_type)
    else:
        raise Exception('Unrecognized input type: ', args.type)


if __name__ == '__main__':
    exr_fn = '/Users/lucy.zhao/Data/mx/cleanGT_exr2raw_expts/graphics_data_various_noise_types/sample_graphics_imgs/GT_3_4200x3200_output_GT_RGB.exr'
    exr_img = cv2.imread(exr_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    exr_img = np.clip(exr_img, 0, 1)
    print('Image bit depth, channel R: ', count_bit_depth(exr_img, channel=2))  # exr is in bgr
    print('Image bit depth, channel G: ', count_bit_depth(exr_img, channel=1))
    print('Image bit depth, channel B: ', count_bit_depth(exr_img, channel=0))
    quantized = tone_map(exr_img, black_level=64, white_level=1023, alg='drago')

    vis = lambda x: (x[:, :, ::-1].astype(np.float32) - 64) / (1023 - 64) * (2 ** -2) ** (1 / 2.2)
    plt.imshow(vis(quantized))
    plt.show()
    quantized = tone_map(exr_img, black_level=64, white_level=1023, alg='he')  # this is very slow
    plt.imshow(vis(quantized))
    plt.show()
    quantized = tone_map(exr_img, black_level=64, white_level=1023, alg='none')
    plt.imshow(vis(quantized))
    plt.show()
