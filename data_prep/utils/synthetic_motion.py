# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
import cv2
import numpy as np


def img_to_burst(img, burst_size, index_ref_frame=0, perturb_ref_frame=False, transformation_params=None, interpolation='bilinear'):
    if interpolation == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError

    burst = []

    for i in range(burst_size):
        if i == index_ref_frame and not perturb_ref_frame:
            image_t = np.copy(img)
        else:
            # Sample random image transformation parameters
            # Translation
            max_translation = transformation_params.get('max_translation', 0.0)
            translation = (random.uniform(-max_translation, max_translation),
                           random.uniform(-max_translation, max_translation))

            # Rotation
            max_rotation = transformation_params.get('max_rotation', 0.0)
            theta = random.uniform(-max_rotation, max_rotation)

            # Shear
            max_shear = transformation_params.get('max_shear', 0.0)
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            shear_factor = (shear_x, shear_y)

            # Anisotropic shear
            max_ar_factor = transformation_params.get('max_ar_factor', 0.0)
            ar_factor = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

            # Scaling
            max_scale = transformation_params.get('max_scale', 0.0)
            scale_factor = np.exp(random.uniform(-max_scale, max_scale))
            scale_factor = (scale_factor, scale_factor * ar_factor)

            output_sz = (img.shape[1], img.shape[0])

            # Generate a affine transformation matrix corresponding to the sampled parameters
            t_mat = get_tmat((img.shape[0], img.shape[1]), translation, theta, shear_factor, scale_factor)

            # Apply the sampled affine transformation
            image_t = cv2.warpAffine(img, t_mat, output_sz, flags=interpolation, borderMode=cv2.BORDER_CONSTANT)

        if transformation_params.get('border_crop') is not None:
            border_crop = transformation_params.get('border_crop')
            image_t = image_t[border_crop:-border_crop, border_crop:-border_crop, :]

        burst.append(image_t)

    burst = np.stack(burst)
    return burst


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)


def torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).numpy()


def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
    """ Generates a transformation matrix corresponding to the input transformation parameters """
    im_h, im_w = image_shape

    t_mat = np.identity(3)

    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

    t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                        [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                        [0.0, 0.0, 1.0]])

    t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                        [0.0, scale_factors[1], 0.0],
                        [0.0, 0.0, 1.0]])

    t_mat = t_scale @ t_rot @ t_shear @ t_mat

    t_mat = t_mat[:2, :]

    return t_mat
