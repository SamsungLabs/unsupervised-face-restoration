"""
Author(s):
Abdelrahman Abdelhamed
"""

import numpy as np


def brightness_transfer(image, dst_image, white_level=1023):
    subsample = 16
    src_image_sample = image[::subsample, ::subsample].flatten()
    dst_image_sample = dst_image[::subsample, ::subsample].flatten()
    bright_func = np.polyfit(src_image_sample, dst_image_sample, 1)
    image_adjusted = image * bright_func[0] + bright_func[1]
    image_adjusted = np.clip(image_adjusted, 0, white_level)
    return image_adjusted


def brightness_transfer_v1(image, dst_image, white_level=1023):
    mean1 = np.mean(image)
    mean2 = np.mean(dst_image)
    std1 = np.std(image)
    std2 = np.std(dst_image)
    std1 = max(std1, 1e-8)
    std2 = max(std2, 1e-8)
    image_adjusted = (image - mean1) / std1 * std2 + mean2
    image_adjusted = np.clip(image_adjusted, 0, white_level)
    return image_adjusted


def brightness_transfer_v11(image, dst_image, white_level=1023):
    mean1 = np.mean(image)
    mean2 = np.mean(dst_image)
    image_adjusted = image - mean1 + mean2
    image_adjusted = np.clip(image_adjusted, 0, white_level)
    return image_adjusted


def brightness_transfer_v2(image, dst_image, white_level=1023):
    subsample = 4
    src_image_sample = image[::subsample, ::subsample].flatten()
    dst_image_sample = dst_image[::subsample, ::subsample].flatten()
    bright_func = np.polyfit(src_image_sample, dst_image_sample, 1)
    image_adjusted = image * bright_func[0] + bright_func[1]
    image_adjusted = np.clip(image_adjusted, 0, white_level)
    return image_adjusted


def standardize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


def normalize(x):
    min_ = np.min(x)
    max_ = np.max(x)
    return (x - min_) / (max_ - min_)
