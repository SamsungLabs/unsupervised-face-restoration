import os
import torch
import numpy as np
import PIL
import torchvision.transforms as transforms
import subprocess
import sys


def save_args(args, file_name):
    """
    Source: https://github.com/VITA-Group/EnlightenGAN/blob/master/options/base_options.py
    EnlightenGAN base_options.py
    """
    args = vars(args)
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

        opt_file.write('\n------------------------------\n')
        opt_file.write(get_git_info())

        opt_file.write('\n------------------------------\n')
        opt_file.write('Shell command:\n')
        opt_file.write(get_command())


def get_git_revision_hash() -> str:
    """
    Source: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    :return:
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_branch() -> str:
    return subprocess.check_output(['git', 'branch']).decode('ascii').strip()


def get_git_info() -> str:
    current_hash = get_git_revision_hash()
    current_branch = get_git_revision_branch()
    git_info = f'Git Info:\nCurrent commit: {current_hash}\nBranches:\n {current_branch}'
    return git_info


def get_command() -> str:
    return " ".join(sys.argv[:])


def get_image_dimensions(img, raise_warning=False):
    # TODO: use this function to replace manual checks wherever possible.
    """
    Returns dimensions of input image or image batch.
    :param img: A 2D, 3D, or 4D numpy.ndarray or torch.Tensor. If the input is a numpy array
    then we assume it is given in the B, H, W, C format. If it is a tensor, we assume it is
    given in the B, C, H, W format.
    :return: H (height), W (width), C (channels), B (batch size)
    Last update: 2020 - 03 - 02 [tsogkas]
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 4:
            B, H, W, C = img.shape
        elif img.ndim == 3:
            H, W, C = img.shape
            B = 1
        elif img.ndim == 2:
            H, W = img.shape
            B = C = 1
        else:
            raise ValueError('Input must be a 2D, 3D, or 4D array')
    elif isinstance(img, torch.Tensor):
        if img.dim() == 4:
            B, C, H, W = img.shape
        elif img.dim() == 3:
            C, H, W = img.shape
            B = 1
        elif img.dim() == 2:
            H, W = img.shape
            B = C = 1
        else:
            raise ValueError('Input must be a 2D, 3D, or 4D tensor')
    else:
        raise TypeError('Input must be a numpy.ndarray or a torch.Tensor')
    if C not in [1, 3] and raise_warning:
        print('WARNING: number of channels should be 1 (grayscale) or 3 (color). '
              'Are you sure this is an image?')
    return H, W, C, B


def is_color_image(img):
    H, W, C, B = get_image_dimensions(img)
    return C == 3 and B == 1


def numpy2tensor(np_arr):
    """
    Converts numpy array to torch tensor, assuming these represent (batches of) images.
    Works with up to 4-D arrays. No normalization or channel changes are performed.
    :param np_arr: (BxHxWxC) or (HxWxC) or (HxW).
    :return: torch_tensor: (BxCxHxW) or (CxHxW) or (HxW)
    Last update: 2020 - 04 - 07
    [tsogkas:] Documentation edit.
    """
    if isinstance(np_arr, torch.Tensor):
        return np_arr
    assert isinstance(np_arr, np.ndarray), 'Input must be a numpy array.'
    if np_arr.ndim == 5:
        np_arr = np.transpose(np_arr, (0, 4, 3, 1, 2))
    elif np_arr.ndim == 4:
        np_arr = np.transpose(np_arr, (0, 3, 1, 2))
    elif np_arr.ndim == 3:
        np_arr = np.transpose(np_arr, (2, 0, 1))
    torch_tensor = torch.from_numpy(np.ascontiguousarray(np_arr))
    return torch_tensor


def tensor2numpy(torch_tensor):
    """
    Converts a torch tensor to a numpy array, assuming these represent (batches of) images.
    Works with up to 4-D arrays. No normalization or channel changes are performed.
    :param torch_tensor: (BxCxHxW) or (CxHxW) or (HxW)
    :return: np_arr: (BxHxWxC) or (HxWxC) or (HxW)
    Last update: 2020 - 04 - 07
    [tsogkas:] Documentation edit.
    """
    if isinstance(torch_tensor, np.ndarray):
        return torch_tensor
    assert isinstance(torch_tensor, torch.Tensor), 'Input must be a torch tensor.'
    torch_tensor = torch_tensor.detach().cpu()
    if torch_tensor.dim() == 5:
        torch_tensor = torch_tensor.permute(0, 1, 3, 4, 2)
    elif torch_tensor.dim() == 4:
        torch_tensor = torch_tensor.permute(0, 2, 3, 1)
    elif torch_tensor.dim() == 3:
        torch_tensor = torch_tensor.permute(1, 2, 0)
    np_arr = torch_tensor.numpy()
    return np_arr


def tensor_img_to_numpy_img(tensor_img, cast_to_uint8=True):
    """
    Converts torch tensor to numpy array, assuming these represent (batches of) images.
    Normalizes in [0,255], changes channel order to RGB, and data type to uint8.
    Works with up to 4-D arrays.
    :param np_arr: (BxHxWxC) or (HxWxC) or (HxW).
    :return: torch_tensor: (BxCxHxW) or (CxHxW) or (HxW)
    Last update: 2020 - 04 - 07
    [tsogkas:] Add documentation.
    """
    assert(in_range(tensor_img, [0, 1]))
    numpy_img = tensor2numpy(tensor_img)
    # Change the channel order: BGR -> RGB
    if numpy_img.ndim in [3, 4]:
        num_channels = numpy_img.shape[2] if numpy_img.ndim == 3 else numpy_img.shape[3]
        if num_channels not in [1, 3]:
            raise ValueError('The number of image channels must be 1 (grayscale) or 3 (color).')
        numpy_img = np.ascontiguousarray(numpy_img[..., ::-1])
    numpy_img = 255 * numpy_img
    if cast_to_uint8:
        numpy_img = numpy_img.astype(np.uint8)
    return numpy_img


def numpy_img_to_tensor_img(numpy_img):
    """
    Converts numpy array to torch tensor, assuming these represent (batches of) images.
    Normalizes in [0,1], changes channel order to RGB, and data type to float32.
    Works with up to 4-D arrays.
    :param np_arr: (BxHxWxC) or (HxWxC) or (HxW).
    :return: torch_tensor: (BxCxHxW) or (CxHxW) or (HxW)
    Last update: 2020 - 04 - 07
    [tsogkas:] Add documentation.
    """
    assert(in_range(numpy_img, [0, 255]))
    # First change the channel order: BGR -> RGB
    if numpy_img.ndim in [3, 4]:
        num_channels = numpy_img.shape[2] if numpy_img.ndim == 3 else numpy_img.shape[3]
        if num_channels not in [1, 3]:
            raise ValueError('The number of image channels must be 1 (grayscale) or 3 (color).')
        numpy_img = numpy_img[..., ::-1]
    tensor_img = numpy2tensor(numpy_img)
    tensor_img = tensor_img.float() / 255.0
    return tensor_img


def in_range(a, val_range, pthresh=1):
    """
    Checks whether val_range[0] <= a[i] <= val_range[1] for all i.
    :param a: input tensor or numpy array.
    :param val_range: 2-element list or tuple with lower an upper bounds of value range
    :param pthresh: (float) between 0 and 1.
    :return: True or False
    """
    # pthresh: percentage of values we allow to be outside the value range
    error_msg = 'Input array is not in the expected value range. {}% of the values are outside {}'
    p = ((a >= val_range[0]) & (a <= val_range[1])).sum()
    if isinstance(a, torch.Tensor):
        p = p.item() / a.nelement()
    else:
        p /= a.size
    if p > pthresh:
        raise ValueError(error_msg.format((1 - p)*100, val_range))
    else:
        return True


def get_freer_gpu():
    """
    Returnes the index of the GPU in the system that has the most RAM available.
    :return: (int)
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def _error_data_type():
    raise TypeError('Input must be a numpy array or a torch tensor.')


def compress_range(img):
    if isinstance(img, np.ndarray):
        # Assume that the value range is in [0, 255.]
        assert in_range(img, [0, 255]), 'Numpy array image values must be in [0, 255].'
        # shift the range to [1, 256] to avoid problems with log()
        img_compressed = np.log(img + 1) / np.log(256.)  # in [0, 1]
        img_compressed *= 255.  # rescale to [0, 255]
    elif isinstance(img, torch.Tensor):
        assert in_range(img, [0, 1]), 'Torch tensor image values must be in [0, 1].'
        # shift the range to [1, 2] to avoid problems with log()
        img_compressed = torch.log(img + 1) / torch.log(torch.tensor(2.))  # in [0, 1]
    else:
        _error_data_type()
    return img_compressed


def invert_image(img):
    return 1 - img if isinstance(img, torch.Tensor) else 255 - img


def imshow(img, invert=False, increase_contrast=False, title='', **kwargs):
    import matplotlib.pyplot as plt
    # If input is of type PIL.Image, convert to torch tensor
    if isinstance(img, PIL.Image.Image):
        img = transforms.ToTensor()(img)
    # we have to first detach and move to cpu()...
    if isinstance(img, torch.Tensor) and img.requires_grad:
        img = img.detach().cpu()
    # ...then apply all the other operations
    if increase_contrast:
        img = compress_range(img)
    if invert:
        img = invert_image(img)
    img = tensor2numpy(img).squeeze() if isinstance(img, torch.Tensor) else img
    plt.imshow(img, **kwargs)
    plt.title(title)
    plt.show()
