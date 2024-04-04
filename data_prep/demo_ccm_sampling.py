"""
Author(s):

Abhijith Punnappurath (abhijith.p@samsung.com)
29 Aug 2022

Apply a random inverse CCM to a graphics image
Two sampling modes are supported : discrete and KNN
"""

import os
import argparse
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        help='path to graphics image',
                        default='C:\Work\Sidi_synth_to_raw/20220531_overall_process_from_exr_to_raw\Garage_DebugBatch1/5/2022_05_21_12_36_52_61e73545-4436-5a25-035c-38b52bcfce38_PathTracer.exr'
                        )
    parser.add_argument('--save_path', type=str,
                        help='path to save output image to',
                        default='C:\Work\Sidi_synth_to_raw\code/results/ccm_sampling_demo/'
                        )
    parser.add_argument('--dictionary_path', type=str,
                        help='path to CCM dictionary',
                        default='C:\Work\Sidi_synth_to_raw\code/ccm_awb_dictionary_52_plus_230.p'
                        )
    parser.add_argument('--sampling_mode', type=str,
                        help='type of ccm sampling. Options: discrete or knn or knn-convex (knn-convex is preferred over knn)',
                        default='knn-convex'
                        )
    parser.add_argument('--k', type=int,
                        help='number of nearest neighbours, applies only for knn sampling_mode',
                        default=3
                        )
    parser.add_argument('--plot_fig',
                        help='plot chromaticity diagram',
                        default=False
                        )

    args = parser.parse_args()

    print(args)

    return args


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def apply_inverse_ccm_on_image(image, forward_ccm):
    forward_ccm = np.reshape(np.asarray(forward_ccm), (3, 3))
    forward_ccm = forward_ccm / np.sum(forward_ccm, axis=1, keepdims=True)
    inverse_ccm = np.linalg.inv(forward_ccm)
    image = inverse_ccm[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
    image = np.sum(image, axis=-1)
    # image = np.clip(image, 0.0, 1.0)
    return image


def apply_color_cast(normalized_image, as_shot_neutral, clip=False):
    as_shot_neutral = 1 / as_shot_neutral
    white_balanced_image = normalized_image / as_shot_neutral
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def get_illum_normalized_by_g(illum_in_arr):
    return illum_in_arr[:, 0] / illum_in_arr[:, 1], illum_in_arr[:, 1] / illum_in_arr[:, 1], illum_in_arr[:,
                                                                                             2] / illum_in_arr[:, 1]


def find_knn_indices(targetillum, allsourceillums, nn_num, show_fig=False):
    targetillum = targetillum[0::2]
    allsourceillums = allsourceillums[:, 0::2]
    mse = allsourceillums - np.tile(targetillum, (allsourceillums.shape[0], 1))
    mse = np.mean(mse * mse, axis=1)
    inds = np.argsort(mse)[0:nn_num]
    weights = mse[inds]

    if show_fig:
        plt.scatter(targetillum[0], targetillum[1], c="pink",
                    linewidths=1,
                    marker="s",
                    edgecolor="green",
                    s=100)

        plt.scatter(allsourceillums[:, 0], allsourceillums[:, 1], c="yellow",
                    linewidths=1,
                    marker="^",
                    edgecolor="red",
                    s=10)

        plt.scatter(allsourceillums[inds, 0], allsourceillums[inds, 1], c="yellow",
                    linewidths=1,
                    marker="o",
                    edgecolor="black",
                    s=100)

    return inds, weights


def compute_knn_ccm(indices, weights, CCMs):
    sampled_ccm = np.zeros([3, 3]).astype('float64')
    iw = 1 / weights
    for i in range(len(indices)):
        sampled_ccm += iw[i] / np.sum(iw) * CCMs[i, :, :].squeeze()

    return sampled_ccm


def main(dictionary_path, sampling_mode, k, exr_img=None, img_path=None, save_path=None, plot_fig=False, apply_ccm=True):
    # either provide a graphics image in the range of [0, 1] or provide a path to the image;
    # img_path is not used if exr_img is provided.
    if exr_img is None:
        assert img_path is not None

    # load CCM dictionary
    data = pickle.load(open(dictionary_path, "rb"))
    CCMs = np.array(data['ccm'])
    illums = 1 / np.array(data['awb'])

    if sampling_mode == 'knn' or sampling_mode == 'knn-convex':
        illums[:, 0], illums[:, 1], illums[:, 2] = get_illum_normalized_by_g(illums)
        illums_mean = np.mean(illums, 0)
        illums_cov = np.cov(np.transpose(illums))

    # read image
    if exr_img is None:
        exr_img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # .astype('float32')
    exr_img = exr_img[:, :, ::-1]  # .exr image is BGR, change it to RGB for processing
    # exr_img = np.clip(exr_img,0,1)

    # sample a CCM matrix
    if sampling_mode == 'discrete':
        idx = np.random.randint(0, CCMs.shape[0])
        sampled_ccm = CCMs[idx]
        sampled_illum = illums[idx]

    elif sampling_mode == 'knn':
        min_r = illums[:, 0].min()
        max_r = illums[:, 0].max()
        min_b = illums[:, 2].min()
        max_b = illums[:, 2].max()
        while True:
            sampled_illum = np.random.multivariate_normal(illums_mean, illums_cov, 1).squeeze()
            if min_r < sampled_illum[0] < max_r and min_b < sampled_illum[2] < max_b:
                break

        indices, weights = find_knn_indices(sampled_illum, illums, k)
        sampled_ccm = compute_knn_ccm(indices, weights, CCMs)

    elif sampling_mode == 'knn-convex':
        while True:
            sampled_illum = np.random.multivariate_normal(illums_mean, illums_cov, 1).squeeze()
            if in_hull(np.expand_dims(sampled_illum[[0, 2]], axis=0), illums[:, [0, 2]]):
                break

        indices, weights = find_knn_indices(sampled_illum, illums, k)
        sampled_ccm = compute_knn_ccm(indices, weights, CCMs)

    else:
        raise Exception('Sampling mode unsupported')

    if plot_fig:
        plt.scatter(illums[:, 0], illums[:, 2], c="pink",
                    linewidths=1,
                    marker="s",
                    edgecolor="green",
                    s=20)
        plt.scatter(sampled_illum[0], sampled_illum[2], c="red",
                    linewidths=2,
                    marker="o",
                    edgecolor="black",
                    s=250)
        plt.show()

    # apply inverse CCM
    raw_img = apply_inverse_ccm_on_image(exr_img, sampled_ccm) if apply_ccm else exr_img
    raw_img_no_wb = apply_color_cast(raw_img, sampled_illum)
    # change results back to BGR
    raw_img = raw_img[:, :, ::-1]
    raw_img_no_wb = raw_img_no_wb[:, :, ::-1]

    # if img_path is not None and save_path is not None:
    #     cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)[:-4] + '.png'),
    #                 ((raw_img ** (1 / 2.2)) * 255).astype('uint8'))
    #     cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)[:-4] + '_no_wb.png'),
    #                 ((raw_img_no_wb ** (1 / 2.2)) * 255).astype('uint8'))
    #     np.save(os.path.join(save_path, os.path.basename(img_path)[:-4] + '_ccm.npy'), sampled_ccm)

    return sampled_ccm, sampled_illum, raw_img_no_wb


if __name__ == "__main__":
    args = parse_args()
    main(args.dictionary_path, args.sampling_mode, args.k, exr_img=None, img_path=args.img_path,
         save_path=args.save_path, plot_fig=args.plot_fig)
