import numpy as np
import copy
from utils.image_utils import stack_bayer, invert_stack_bayer


def synthesize_noisy_image_ITS(clean, inverse_transforms):
    clean = stack_bayer(clean)
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
        bin_ranges = np.append(bin_ranges, 1023)
        bin_ranges = np.insert(bin_ranges, 0, 0)

        for b in range(bin_ranges.shape[0] - 1):
            bmin = bin_ranges[b]
            bmax = bin_ranges[b + 1]
            ind = np.where(np.logical_and(clean[:, ch] >= bmin, clean[:, ch] <= bmax))
            n_samples = ind[0].shape[0]
            r = np.random.rand(n_samples)
            inv_cdf = inverse_transforms[ch][b]
            r = inv_cdf(r)
            noisy[ind, ch] = noisy[ind, ch] + r

    noisy = noisy.reshape(cleanshape)
    noisy = invert_stack_bayer(noisy)
    return noisy
