# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import numpy as np


def norm_01(img):
    """ Normalize img to the range [0, 1], inclusive.

    Args:
        img (np.ndarray): image to normalize, can be any dtype.

    Returns:
        np.ndarray: normalized image, dtype=np.float64
    """

    img_out = img.astype(np.float64)
    img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min())

    return img_out


def norm_ab(img, a, b):
    """ Normalize img to the range [a, b], inclusive. Float output by default.

    If a < b, then the histogram will just be scaled and shifted.
    If b > a, then the histogram will be flipped left-right, scaled, and shifted.

    Args:
        img (np.ndarray): image to normalize, can be any dtype.
        a (float): one end of range
        b (float): other end of range

    Returns:
        np.ndarray: normalized image, dtype=np.float64
    """

    img_out = norm_01(img)
    # scale to the extent of the range, then shift to match
    img_out = (img_out * (b - a)) + a

    return img_out


def norm_dtype(img, dtype=np.uint8):
    """ Normalize img to the range [dtype_min, dtype_max].

    Can be used to prepare images for file output.
    Equivalent to a 0% histogram stretch.
    Works by converting to float64, then stretching/shifting, then quantizing.

    Args:
        img (np.ndarray): image to normalize, can be any dtype.
        dtype (np dtype): integer datatype to normalize to.

    Returns:
        np.ndarray: normalized image, dtype=dtype
    """
    dtype_info = np.iinfo(dtype)

    return norm_ab(img, dtype_info.min, dtype_info.max).astype(dtype)
