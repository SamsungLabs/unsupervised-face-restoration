import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import copy
import shutil
from noise_profiler.image_synthesizer import synthesize_noisy_image_v2

BLACK_LEVEL = 64
WHITE_LEVEL = 1023


def change_blacklevel(img, curr_bl=0, new_bl=64):
    # TODO: should I actually use this or a simpler version? Ask Ali.
    """Change the black level of the given image.

    :param numpy img: Image to be processed
    :param int curr_bl: The current black level of the given image
    :param int new_bl: The new black level to applied on the image

    return numpy img: Image with the new black level value.
    """
    img_max = img.max()
    img = (img - curr_bl) / (img_max - curr_bl)
    new_whitepoint = max((img_max - new_bl), new_bl)
    img = (img * new_whitepoint) + new_bl

    return img


def BGR2TetraRAW(img_bgr, bayer_type=2):
    """Convert a bgr image to a bayer GRBG if bayer_type == 1 and GBRG if bayer_type == 2

    :param numpy.array img_bgr: A numpy array image in bgr with shape HxWxC

    :return numpy.array bayer: A numpy array image in bayer GRBG with shape HxW
    """
    h, w = img_bgr.shape[:2]
    bayer = np.empty((h, w), dtype=np.float32)

    (B, G, R) = cv2.split(img_bgr)

    if bayer_type in ['1', 'GRBG']:
        # G R B G
        bayer[0::4, 0::4] = G[0::4, 0::4]  # top left
        bayer[0::4, 1::4] = G[0::4, 1::4]  # top right
        bayer[1::4, 0::4] = G[1::4, 0::4]  # bottom left
        bayer[1::4, 1::4] = G[1::4, 1::4]  # bottom right

        bayer[0::4, 2::4] = R[0::4, 2::4]  # top left
        bayer[0::4, 3::4] = R[0::4, 3::4]  # top right
        bayer[1::4, 2::4] = R[1::4, 2::4]  # bottom left
        bayer[1::4, 3::4] = R[1::4, 3::4]  # bottom right

        bayer[2::4, 0::4] = B[2::4, 0::4]  # top left
        bayer[2::4, 1::4] = B[2::4, 1::4]  # top right
        bayer[3::4, 0::4] = B[3::4, 0::4]  # bottom left
        bayer[3::4, 1::4] = B[3::4, 1::4]  # bottom right

        bayer[2::4, 2::4] = G[2::4, 2::4]  # top left
        bayer[2::4, 3::4] = G[2::4, 3::4]  # top right
        bayer[3::4, 2::4] = G[3::4, 2::4]  # bottom left
        bayer[3::4, 3::4] = G[3::4, 3::4]  # bottom right
    elif bayer_type in ['2', 'GBRG']:
        # G B R G
        bayer[0::4, 0::4] = G[0::4, 0::4]  # top left
        bayer[0::4, 1::4] = G[0::4, 1::4]  # top right
        bayer[1::4, 0::4] = G[1::4, 0::4]  # bottom left
        bayer[1::4, 1::4] = G[1::4, 1::4]  # bottom right

        bayer[0::4, 2::4] = B[0::4, 2::4]  # top left
        bayer[0::4, 3::4] = B[0::4, 3::4]  # top right
        bayer[1::4, 2::4] = B[1::4, 2::4]  # bottom left
        bayer[1::4, 3::4] = B[1::4, 3::4]  # bottom right

        bayer[2::4, 0::4] = R[2::4, 0::4]  # top left
        bayer[2::4, 1::4] = R[2::4, 1::4]  # top right
        bayer[3::4, 0::4] = R[3::4, 0::4]  # bottom left
        bayer[3::4, 1::4] = R[3::4, 1::4]  # bottom right

        bayer[2::4, 2::4] = G[2::4, 2::4]  # top left
        bayer[2::4, 3::4] = G[2::4, 3::4]  # top right
        bayer[3::4, 2::4] = G[3::4, 2::4]  # bottom left
        bayer[3::4, 3::4] = G[3::4, 3::4]  # bottom right

    return bayer


def BGR2BayerRAW(img_bgr, bayer_type=2):
    """Convert a bgr image to a bayer GRBG if bayer_type == 1 and GBRG if bayer_type == 2

    :param numpy.array img_bgr: A numpy array image in bgr with shape HxWxC

    :return numpy.array bayer: A numpy array image in bayer GRBG with shape HxW
    """
    h, w = img_bgr.shape[:2]
    bayer = np.empty((h, w), dtype=np.float32)

    (B, G, R) = cv2.split(img_bgr)

    if bayer_type == '1' or bayer_type == 'GRBG':
        # G R B G
        bayer[0::2, 0::2] = G[0::2, 0::2]  # top left
        bayer[0::2, 1::2] = R[0::2, 1::2]  # top right
        bayer[1::2, 0::2] = B[1::2, 0::2]  # bottom left
        bayer[1::2, 1::2] = G[1::2, 1::2]  # bottom right
    elif bayer_type == '2' or bayer_type == 'GBRG':
        # G B R G
        bayer[0::2, 0::2] = G[0::2, 0::2]  # top left
        bayer[0::2, 1::2] = B[0::2, 1::2]  # top right
        bayer[1::2, 0::2] = R[1::2, 0::2]  # bottom left
        bayer[1::2, 1::2] = G[1::2, 1::2]  # bottom right

    return bayer


def stack_raw_image(image, tile_size):
    '''' stack Bayer format to a 4-channel array
         tile size is 2 for tetra binary CFA and it is 1 for regular bayer CFA
         h x w -> h/2 x w/2 x c
    '''

    assert image.shape[0] % tile_size == 0
    assert image.shape[1] % tile_size == 0
    assert len(image.shape) == 2

    height, width = image.shape

    h = height // 2
    w = width // 2

    stacked_raw = np.zeros((h, w, 4))

    chans = [[0, 0], [0, tile_size], [tile_size, 0], [tile_size, tile_size]]

    for c in range(4):
        mask_ch = np.zeros((height, width), dtype=np.bool_)
        for t_row in range(tile_size):
            for t_col in range(tile_size):

                mask_ch[chans[c][0]+t_row::tile_size*2, chans[c][1]+t_col::tile_size*2] = True
        stacked_raw[:, :, c] = image[np.where(mask_ch)].reshape(h, w)

    return stacked_raw


def pack_stacked_raw(stacked_raw, tile_size):
    '''' Pack a stacked (4-channel) raw image into regular Bayer format
         tile size is 2 for tetra binary CFA and it is 1 for regular bayer CFA
         h/2 x w/2 x c -> h x w
    '''

    assert len(stacked_raw.shape) == 3

    height, width, _ = stacked_raw.shape  #

    h = height * 2
    w = width * 2

    packed_raw = np.zeros((h,w))

    chans = [[0,0],[0,tile_size],[tile_size,0],[tile_size,tile_size]]

    for c in range(4):
        mask_ch = np.zeros((h, w), dtype=np.bool_)
        for t_row in range(tile_size):
            for t_col in range(tile_size):

                mask_ch[chans[c][0]+t_row::tile_size*2,chans[c][1]+t_col::tile_size*2]=True

        packed_raw[np.where(mask_ch)] = stacked_raw[:, :, c].flatten()

    return packed_raw


def add_noise_to_tetra_raw(raw, noise_model, iso):
    if isinstance(noise_model, dict) and 'noise_model' in noise_model.keys():
        noisy_raw = synthesize_noisy_image_v2(raw, model=noise_model['noise_model'],
                                          dst_iso=iso,
                                          min_val=0, max_val=WHITE_LEVEL,
                                          iso2b1_interp_splines=noise_model['iso2b1_interp_splines'],
                                          iso2b2_interp_splines=noise_model['iso2b2_interp_splines'])
    elif isinstance(noise_model, list):
        noisy_raw = synthesize_noisy_image_ITS(raw, noise_model).clip(0, WHITE_LEVEL)
    else:
        raise ValueError('Invalid noise model type.')
    return noisy_raw


def add_noise_to_bayer_raw(raw, noise_model, iso):
    noisy_raw = synthesize_noisy_image_v2(raw, model=noise_model['noise_model'],
                                      dst_iso=iso,
                                      min_val=0, max_val=WHITE_LEVEL,
                                      iso2b1_interp_splines=noise_model['iso2b1_interp_splines'],
                                      iso2b2_interp_splines=noise_model['iso2b2_interp_splines'])
    return noisy_raw


def synthesize_noisy_image_ITS(clean, inverse_transforms, sensor_type="s22+tetra", black_level=1023):
    tile_size = 2 if sensor_type == "s22+tetra" else 1
    clean = stack_raw_image(clean, tile_size=tile_size)

    cleanshape = clean.shape
    clean = clean.reshape((-1, 4))
    noisy = copy.deepcopy(clean)
    noisy = noisy.astype(np.float32)

    for ch in range(4):
        if np.where(inverse_transforms[-1] == -1)[0].size == 1:  # there should be some simpler check than this
            mid_vals = inverse_transforms[ch][-1]
        else:
            mid_vals = inverse_transforms[-1]
        bin_ranges = np.diff(mid_vals) / 2 + mid_vals[:-1]
        bin_ranges = np.append(bin_ranges, black_level)
        bin_ranges = np.insert(bin_ranges, 0, -1)

        idx = np.digitize(clean[:, ch], bin_ranges, right=True)

        for b in range(bin_ranges.shape[0] - 1):
            bmin = bin_ranges[b]
            bmax = bin_ranges[b + 1]
            ind = np.where(idx == b + 1)
            n_samples = ind[0].shape[0]
            r = np.random.rand(n_samples)
            inv_cdf = inverse_transforms[ch][b]
            r = inv_cdf(r)
            noisy[ind, ch] = noisy[ind, ch] + r

    noisy = noisy.reshape(cleanshape)
    noisy = pack_stacked_raw(noisy, tile_size)
    return noisy


def exr_to_raw(exr_img, cfa_pattern, bayer_type, ev):
    """
    Assumes BGR format for exr_img
    """
    def change_exposure(img, EV):
        return img * (2 ** EV)

    exr_img = change_exposure(exr_img, ev)
    exr_img = np.clip(exr_img, 0, 1)

    if cfa_pattern == 'tetra':
        raw = BGR2TetraRAW(exr_img, bayer_type)
    else:
        raw = BGR2BayerRAW(exr_img, bayer_type)

    raw = raw * WHITE_LEVEL
    raw = change_blacklevel(raw, new_bl=BLACK_LEVEL)  # TODO: replace this with Ali's function?
    raw = np.clip(raw, 0, WHITE_LEVEL)
    return raw


def process_tetra_raw_for_visualization(raw, bayer_type):
    raw_png_bgr = np.tile(np.expand_dims(raw, 2), (1, 1, 3))

    # !!!CAUTION!!! Image is BGR format (cv2.imread)
    mask = np.zeros_like(raw_png_bgr)

    if bayer_type == '1' or bayer_type == 'GRBG':
        # Red channel
        mask[0::4, 2::4, 2] = 1
        mask[0::4, 3::4, 2] = 1
        mask[1::4, 2::4, 2] = 1
        mask[1::4, 3::4, 2] = 1

        # Green channel
        mask[0::4, 0::4, 1] = 1
        mask[0::4, 1::4, 1] = 1
        mask[1::4, 0::4, 1] = 1
        mask[1::4, 1::4, 1] = 1
        mask[2::4, 2::4, 1] = 1
        mask[2::4, 3::4, 1] = 1
        mask[3::4, 2::4, 1] = 1
        mask[3::4, 3::4, 1] = 1

        # Blue channel
        mask[2::4, 0::4, 0] = 1
        mask[2::4, 1::4, 0] = 1
        mask[3::4, 0::4, 0] = 1
        mask[3::4, 1::4, 0] = 1

    elif bayer_type == '2' or bayer_type == 'GBRG':

        # Blue channel
        mask[0::4, 2::4, 0] = 1
        mask[0::4, 3::4, 0] = 1
        mask[1::4, 2::4, 0] = 1
        mask[1::4, 3::4, 0] = 1

        # Green channel
        mask[0::4, 0::4, 1] = 1
        mask[0::4, 1::4, 1] = 1
        mask[1::4, 0::4, 1] = 1
        mask[1::4, 1::4, 1] = 1
        mask[2::4, 2::4, 1] = 1
        mask[2::4, 3::4, 1] = 1
        mask[3::4, 2::4, 1] = 1
        mask[3::4, 3::4, 1] = 1

        # Red channel
        mask[2::4, 0::4, 2] = 1
        mask[2::4, 1::4, 2] = 1
        mask[3::4, 0::4, 2] = 1
        mask[3::4, 1::4, 2] = 1
    else:
        raise ValueError

    raw_png_bgr *= mask

    return raw_png_bgr


def process_bayer_raw_for_visualization(raw, bayer_type):
    # Assumes BGR forma for input and output
    raw_png_bgr = np.tile(np.expand_dims(raw, 2), (1, 1, 3))
    mask = np.zeros_like(raw_png_bgr)

    if bayer_type == '1' or bayer_type == 'GRBG':
        mask[0::2, 0::2, 1] = 1  # top left (G)
        mask[0::2, 1::2, 2] = 1  # top right (R)
        mask[1::2, 0::2, 0] = 1  # bottom left (B)
        mask[1::2, 1::2, 1] = 1  # bottom right (G)
    elif bayer_type == '2' or bayer_type == 'GBRG':
        mask[0::2, 0::2, 1] = 1  # top left (G)
        mask[0::2, 1::2, 0] = 1  # top right (B)
        mask[1::2, 0::2, 2] = 1  # bottom left (R)
        mask[1::2, 1::2, 1] = 1  # bottom right (G)
    else:
        raise ValueError
    raw_png_bgr *= mask

    return raw_png_bgr
