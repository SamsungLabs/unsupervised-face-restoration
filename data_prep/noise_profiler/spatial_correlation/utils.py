"""
Author(s):
Abdelrahman Abdelhamed
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from noise_profiler.img_utils import normalize


def sample_norm(h_, w_):
    return np.random.normal(0, 1, (h_, w_))


def sample_cov(cov_mat_, h_, w_):
    n_var = 4
    sample_cov_ = np.random.multivariate_normal([0] * n_var, cov_mat_, h_ * w_ // n_var)
    sample_cov_image_ = np.zeros((h_, w_))
    sample_cov_image_[0::2, 0::2] = sample_cov_[:, 0].reshape((h_ // 2, w_ // 2))
    sample_cov_image_[0::2, 1::2] = sample_cov_[:, 1].reshape((h_ // 2, w_ // 2))
    sample_cov_image_[1::2, 0::2] = sample_cov_[:, 2].reshape((h_ // 2, w_ // 2))
    sample_cov_image_[1::2, 1::2] = sample_cov_[:, 3].reshape((h_ // 2, w_ // 2))
    return sample_cov_image_, sample_cov_


# save images
def save_image(im, save_fn, sc=4):
    cv2.imwrite(save_fn, cv2.resize((normalize(im) * 255).astype(np.uint8), dsize=(im.shape[1] * sc, im.shape[0] * sc),
                                    interpolation=cv2.INTER_NEAREST))


# plot covariance matrix as a heat map
def plot_cov_mat(cov_mat_, save_fn):
    fig = plt.figure()
    plt.imshow(cov_mat_)
    ax = plt.gca()
    plt.colorbar()
    plt.clim(0, 1)
    for i in range(4):
        for j in range(4):
            ax.annotate("{:.4f}".format(cov_mat_[i, j]), xy=(i - .25, j))
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True)
    plt.xticks(np.arange(0, 4), labels=['Gr', 'R', 'B', 'Gb'])
    plt.yticks(np.arange(0, 4), labels=['Gr', 'R', 'B', 'Gb'])
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_fn)
    plt.close(fig)
