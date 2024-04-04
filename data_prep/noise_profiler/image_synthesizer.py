"""
Author(s):
Abdelrahman Abdelhamed
"""
import os
import numpy as np

from noise_profiler.img_utils import brightness_transfer
from noise_profiler.spatial_correlation.utils import sample_cov


def synthesize_noisy_image(src_image, metadata, model, src_iso, dst_iso, min_val, max_val, dst_image=None, debug=False,
                           per_channel=False, fix_exp=None, match_bright_gt=True, spat_var=False, cross_corr=False,
                           cov_mat_fn=None, iso2b1_interp_splines=None, iso2b2_interp_splines=None):
    """
    Synthesize a high-ISO image with `dst_iso` from a lowISO one `src_image` with `src_iso`.
    Using a heteroscedastic Gaussian noise model `model`.
    :param src_image: Clean (ISO-50) image.
    :param metadata: Image metadata.
    :param model: Noise model.
    :param src_iso: ISO of `src_image`.
    :param dst_iso: ISO of noisy image to be synthesized.
    :param min_val: Minimum image intensity.
    :param max_val: Maximum image intensity.
    :param dst_image: If not None, match the brightness with provided destination image `dst_image`.
    :param debug: Whether to perform some debugging steps.
    :param per_channel: Whether to apply model per channel.
    :param fix_exp: Whether to fix image exposure before and after noise synthesis.
    :param match_bright_gt: Optionally, match the brightness with provided source image `src_image`.
    :param spat_var: Simulate spatial variations in noise.
    :param cross_corr: Simulate spatial cross correlations of noise.
    :param cov_mat_fn: Filename of covariance matrix used to simulate spatial correlations.
    :param iso2b1_interp_splines: Interpolation/extrapolation splines for shot noise (beta1)
    :param iso2b2_interp_splines: Interpolation/extrapolation splines for read noise (beta2)
    :return: Synthesized noisy image, and optionally, a another copy of the image with brightness matching `dst_image`.
    """
    # make a copy
    image = src_image.copy().astype(np.float32)

    # brightness transfer using a target image
    if dst_image is not None:
        image_adjusted = brightness_transfer(image, dst_image)
    else:
        image_adjusted = None

    # apply lens shading correction
    # image_lsc = apply_lens_gain(image, metadata['lens_gain_resized'], metadata['cfa_pattern'],
    #                             operation='multiply')
    # image_lsc_clip = np.clip(image_lsc, min_val, max_val)

    # TODO: account for noise already in image
    # src_params = model[src_iso]

    if fix_exp is not None:
        # update image exposure to match target ISO before synthesizing noise, then re-scale it back later
        image_fix_exp = np.round(image.astype(np.float32) * fix_exp).astype(np.uint32)
    else:
        image_fix_exp = None

    # get noise params (shot, read), per channel
    dst_params = get_noise_params(model, iso2b1_interp_splines, iso2b2_interp_splines, dst_iso)

    # compute noise variance, std. dev.
    if per_channel:
        dst_var = np.zeros(shape=image.shape)
        bp_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for ch in range(4):
            i0 = bp_idx[ch][0]
            j0 = bp_idx[ch][1]
            if fix_exp is not None:
                dst_var[i0::2, j0::2] = image_fix_exp[i0::2, j0::2] * dst_params[ch, 0] + dst_params[ch, 1]
            else:
                dst_var[i0::2, j0::2] = image[i0::2, j0::2] * dst_params[ch, 0] + dst_params[ch, 1]
    else:
        dst_var = image * dst_params[0] + dst_params[1]

    # simulate variance of noise variance
    if spat_var:
        dst_var[dst_var < 0] = 0
        dst_var += np.random.normal(loc=0, scale=1, size=image.shape) * np.sqrt(dst_var)

    dst_var[dst_var < 0] = 0

    # std. dev.
    dst_std = np.sqrt(dst_var)

    if cross_corr:
        # Simplex noise, with cross correlations
        # noise = generate_simplex_noise_template_tiles(image.shape[0], image.shape[1], scale=2)
        # sample noise from estimated covariance matrix
        cov_mat = np.load(cov_mat_fn)
        noise = sample_cov(cov_mat, image.shape[0], image.shape[1])
        noise = noise[0]
    else:
        # Normal Gaussian noise
        noise = np.random.normal(loc=0, scale=1, size=image.shape)

    # scale by heteroscedastic standard deviation
    noise *= dst_std

    if fix_exp is not None:
        noise /= fix_exp

    # invert lens shading correction for noise only
    # noise_inv_lsc = apply_lens_gain(noise, metadata['lens_gain_resized'], metadata['cfa_pattern'],
    #                                 operation='divide')

    # add noise
    noisy_image = (image + noise).astype(image.dtype)

    # invert lens shading correction for noisy image
    # noisy_image = apply_lens_gain(noisy_image, metadata['lens_gain_resized'], metadata['cfa_pattern'],
    #                               operation='divide')

    # defective pixels
    noisy_image = synthesize_defective_pixels(noisy_image)

    # match brightness of source image
    if match_bright_gt:
        noisy_image = brightness_transfer(noisy_image, src_image)

    # clip
    noisy_image = np.clip(noisy_image, min_val, max_val)

    # apply noise on image with adjusted brightness
    noisy_image_adjusted = None
    if dst_image is not None:
        if per_channel:
            dst_var_adj = np.zeros(shape=image_adjusted.shape)
            bp_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for ch in range(4):
                i0 = bp_idx[ch][0]
                j0 = bp_idx[ch][1]
                dst_var_adj[i0::2, j0::2] = image_adjusted[i0::2, j0::2] * dst_params[ch, 0] + dst_params[ch, 1]
        else:
            dst_var_adj = image_adjusted * dst_params[0] + dst_params[1]

        # simulate variance of noise variance
        if spat_var:
            dst_var_adj[dst_var_adj < 0] = 0
            dst_var_adj += np.random.normal(loc=0, scale=1, size=image.shape) * np.sqrt(dst_var_adj)

        dst_var_adj[dst_var_adj < 0] = 0

        dst_std_adj = np.sqrt(dst_var_adj)
        noise_adj = np.random.normal(loc=0, scale=1, size=image_adjusted.shape)
        noise_adj *= dst_std_adj
        noisy_image_adjusted = image_adjusted + noise_adj
        noisy_image_adjusted = np.clip(noisy_image_adjusted, min_val, max_val)

    # results
    synth_res = dict()
    synth_res['noisy_image'] = noisy_image
    synth_res['noisy_image_adjusted'] = noisy_image_adjusted
    return synth_res


def synthesize_noisy_image_v2(src_image, model, metadata=None, src_iso=None, min_val=None, max_val=None, dst_iso=None,
                              dst_image=None, per_channel=True, fix_exp=None, match_bright_gt=True, spat_var=False,
                              cross_corr=False, cov_mat_fn=None, iso2b1_interp_splines=None,
                              iso2b2_interp_splines=None,
                              debug=False, black_diff=0):
    """
    Synthesize a noisy image from `src_image` using a heteroscedastic Gaussian noise model `model`.
    :param src_image: Clean/Semi-clean image.
    :param metadata: Image metadata.
    :param model: Noise model.
    :param src_iso: ISO of `src_image`.
    :param dst_iso: ISO of noisy image to be synthesized.
    :param min_val: Minimum image intensity.
    :param max_val: Maximum image intensity.
    :param dst_image: If not None, match the brightness with provided destination image `dst_image`.
    :param debug: Whether to perform some debugging steps.
    :param per_channel: Whether to apply model per channel.
    :param fix_exp: Whether to fix image exposure before and after noise synthesis.
    :param match_bright_gt: Optionally, match the brightness with provided source image `src_image`.
    :param spat_var: Simulate spatial variations in noise.
    :param cross_corr: Simulate spatial cross correlations of noise.
    :param cov_mat_fn: Filename of covariance matrix used to simulate spatial correlations.
    :param iso2b1_interp_splines: Interpolation/extrapolation splines for shot noise (beta1)
    :param iso2b2_interp_splines: Interpolation/extrapolation splines for read noise (beta2)
    :param black_diff: Difference in black level from the original image and the synthesized noisy image.
        (e.g., original_black - synthetic_black).
    :return: Synthesized noisy image, and optionally, a another copy of the image with brightness matching `dst_image`.
    """
    # make a copy
    image = src_image.copy().astype(np.float32)

    if fix_exp is not None:
        # update image exposure to match target ISO before synthesizing noise, then re-scale it back later
        image_fix_exp = np.round(image.astype(np.float) * fix_exp).astype(np.uint32)
    else:
        image_fix_exp = None

    # if target ISO is not specified, select a random value
    if dst_iso is None:
        dst_iso = np.random.randint(50, 3201)

    if iso2b1_interp_splines is None or iso2b2_interp_splines is None:
        iso2b1_interp_splines = model['iso2b1_interp_splines']
        iso2b2_interp_splines = model['iso2b2_interp_splines']

    # get noise params (shot, read), per channel
    if dst_iso in model:
        dst_params = model[dst_iso]
    else:
        dst_params = np.zeros((4, 2))
        for c in range(4):
            dst_params[c, 0] = iso2b1_interp_splines[c](dst_iso)
            dst_params[c, 1] = iso2b2_interp_splines[c](dst_iso)

    # compute noise variance, std. dev.
    # TODO change this to work for 3 channels
    if per_channel:
        dst_var = np.zeros(shape=image.shape)
        bp_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for ch in range(4):
            i0 = bp_idx[ch][0]
            j0 = bp_idx[ch][1]
            if fix_exp is not None:
                dst_var[i0::2, j0::2] = image_fix_exp[i0::2, j0::2] * dst_params[ch, 0] + dst_params[ch, 1]
            else:
                # Fix: account for difference in black level between original image and synthetic noisy image
                dst_var[i0::2, j0::2] = (image[i0::2, j0::2] + black_diff) * dst_params[ch, 0] + dst_params[ch, 1]
    else:
        dst_var = image * dst_params[0] + dst_params[1]

    # simulate variance of noise variance
    if spat_var:
        dst_var[dst_var < 0] = 0
        dst_var += np.random.normal(loc=0, scale=1, size=image.shape) * np.sqrt(dst_var)

    dst_var[dst_var < 0] = 0

    # std. dev.
    dst_std = np.sqrt(dst_var)

    # Normal Gaussian noise
    noise = np.random.normal(loc=0, scale=1, size=image.shape)

    # scale by heteroscedastic standard deviation
    noise *= dst_std

    if fix_exp is not None:
        noise /= fix_exp

    # add noise
    noisy_image = (image + noise).astype(image.dtype)

    # clip
    noisy_image = np.clip(noisy_image, min_val, max_val)

    return noisy_image


def synthesize_defective_pixels(image, min_def_perc=0.001, max_def_perc=0.01):
    def_pix_perc = np.random.uniform(min_def_perc, max_def_perc)
    n_def_pix = int(def_pix_perc * image.shape[0] * image.shape[1])
    y_def = np.random.uniform(0, image.shape[0], size=n_def_pix).astype(np.int32)
    x_def = np.random.uniform(0, image.shape[1], size=n_def_pix).astype(np.int32)
    def_multiplier = 1.1
    def_std = .05
    image[y_def, x_def] *= np.random.normal(def_multiplier, def_std, size=n_def_pix)
    return image


def load_noise_model(path):
    """
    Load noise model.
    :param path: Path to noise model: either a directory to a non-mixture model or a file path to a mixture model.
    :return: Corresponding noise parameter data.
    """
    if os.path.isdir(path):
        noise_model_path = os.path.join(path, 'model_params.npy')
        iso2b1_interp_splines_fn = os.path.join(path, 'iso2b1_interp_splines.npy')
        iso2b2_interp_splines_fn = os.path.join(path, 'iso2b2_interp_splines.npy')
        return load_model(noise_model_path, iso2b1_interp_splines_fn, iso2b2_interp_splines_fn)
    else:
        return load_noise_mixture_model(path)


def load_model(noise_model_path, iso2b1_interp_splines_fn, iso2b2_interp_splines_fn):
    """
    Load noise model files.
    :param noise_model_path: Path to model directory.
    :param iso2b1_interp_splines_fn: Shot noise interpolation spline file name.
    :param iso2b2_interp_splines_fn: Read noise interpolation spline file name.
    :return: Noise discrete parameters, shot noise interpolation spline, read noise interpolation spline.
    """
    model_arr = np.load(noise_model_path)
    model_ = dict()
    for im_idx in range(model_arr.shape[0] // 4):
        model_[model_arr[im_idx * 4, 0]] = model_arr[im_idx * 4:(im_idx + 1) * 4, 2:4]
    # load interpolation/extrapolation splines for noise parameters
    iso2b1_interp_splines = np.load(iso2b1_interp_splines_fn, allow_pickle=True)
    iso2b2_interp_splines = np.load(iso2b2_interp_splines_fn, allow_pickle=True)
    return model_, iso2b1_interp_splines, iso2b2_interp_splines


def load_noise_mixture_model(model_path):
    """
    Load a noise mixture model.
    :param model_path: Path to model file.
    :return: Noise mixture model (dictionary).
    """
    return np.load(model_path, allow_pickle=True)[()]


def get_noise_params(model, iso2b1_interp_splines, iso2b2_interp_splines, dst_iso):
    """
    Get noise parameters for a given ISO from a noise model.
    :param model: Noise model (single or mixture).
    :param iso2b1_interp_splines: Shot noise interpolation spline.
    :param iso2b2_interp_splines: Read noise interpolation spline.
    :param dst_iso: Target ISO.
    :return: Noise parameter for the given ISO: read and shot noise per channel.
    """
    if 'shot' in model:  # this is a mixture model
        noise_params = get_noise_params_mixture_model(model, dst_iso)
    else:  # non-mixture model
        noise_params = get_noise_params_interp(iso2b1_interp_splines, iso2b2_interp_splines, dst_iso)
    return noise_params


def get_noise_params_interp(iso2b1_interp_splines, iso2b2_interp_splines, dst_iso):
    """
    Get noise parameters (shot, read) per channel, for a given ISO.
    :param iso2b1_interp_splines: Shot noise interpolation spline.
    :param iso2b2_interp_splines: Read noise interpolation spline.
    :param dst_iso: Target ISO level.
    :return: Noise parameters.
    """
    noise_params = np.zeros((4, 2))
    for c in range(4):
        noise_params[c, 0] = iso2b1_interp_splines[c](dst_iso)
        noise_params[c, 1] = iso2b2_interp_splines[c](dst_iso)
    return noise_params


def get_noise_params_mixture_model(model, dst_iso):
    """
    Get noise parameters from a noise mixture model for a given ISO.
    :param model: Noise mixture model.
    :param dst_iso: Target ISO.
    :return: Noise parameter for the given ISO: read and shot noise per channel.
    """
    n_componets = len(model['shot'])
    mixture_idx = np.random.choice(n_componets, size=1, replace=True, p=model['weight'])[0]
    return get_noise_params_interp(model['shot'][mixture_idx], model['read'][mixture_idx], dst_iso)
