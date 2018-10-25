from __future__ import absolute_import
import numpy as np
from collections import Iterable
from .constants import CV2_INTERPOLATION_TYPES
from .constants import NUMPY_TYPES
from .Exceptions import InvalidInterpolationType
from .Exceptions import InvalidNumpyType
# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
"""
Helper functions that contain canned tests or checks that we will run
frequently
"""


def interpolation_type_check(interp):
    """
    checks to see if the interpolation type is one of the acceptable
    values specified in opencv, otherwise raises an
    imsciutils.InvalidInterpolationType error
    """
    if interp not in CV2_INTERPOLATION_TYPES:
        raise InvalidInterpolationType(interp)

    return True


def dtype_type_check(dtype):
    """
    checks to see if the interpolation type is one of the acceptable
    values specified in opencv, otherwise raises an
    imsciutils.InvalidInterpolationType error
    """
    if dtype not in NUMPY_TYPES:
        raise InvalidNumpyType(interp)

    return True


def is_numpy_array(arr):
    """returns True if input is a numpy array or subclass of numpy array"""
    return isinstance(arr,np.ndarray)

def is_iterable(var):
    """returns True if input is an iterable type, false otherwise"""
    return isinstance(var,Iterable)



def type_error_message(variable,variable_name,desired_types):
    """
    returns an error message for a type error_msg

    Args:
        variable (the variable you want raise an error for):
        variable_name (the name of variable):
        desired_types (type,iterable of desired types):

    Returns:
        error_msg (str): formatted error message string
    """
    if not isinstance(desired_types,Iterable):
        desired_types = [desired_types]
    error_msg =  "'{name}' must be on of [{desired}], currently is {cur}"\
        .format(
            name=variable_name,
            desired=','.join(desired_types),
            cur=type(variable))
    return error_msg
