# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import numpy as np
from .printout import error as iuerror

class Summarizer(dict):
    """
    Summarization object for numpy array. The primary job of this
    object is to streamline and simply printing out numpy array objects
    which normally appear as a stream of barely comprehendable data

    This dictionary subclass will return the following when printed out
    or otherwise stringified

    Args:
        input_array (np.ndarray): input array to summarize

    Attributes:
        input_array: original numpy array this object is summarizing
        last_summary: last calculated summary dictionary
                contains the following: shape, size, max, min, mean, dtype
        last_string: last representation string calculated for this array


    Example:
        >>> import imsciutils as iu
        >>> a = np.random.rand(512,512)
        >>> a = iu.Summarizer(a)
        >>> print(a)
        [ARRAY SUMMARY | shape: (512, 512) | size: 262144 | max: 1.0 | min: 0.0 | mean: 0.5 | dtype: float64]
    """
    def __init__(self, input_array):
        """Instantiations function

        Args:
            input_array (np.ndarray): input array to summarize

        """
        # ERROR CHECKING
        if not isinstance(input_array, np.ndarray):
            error_msg = "'input_array' input must be a np.ndarray"
            iuerror(error_msg)
            raise TypeError(error_msg)
        # END ERROR CHECKING
        self.input_array = input_array
        self.last_summary = None
        self.last_string = None
        self.__update()

    def __str__(self):
        """returns a stringified summary"""
        self.__update()
        return self.last_string

    def summarize(self):
        """returns an output dictionary of array attributes

        Args:
            None

        Returns:
            summary (dict): dict containing the following [shape, size, max,
                min, mean, dtype]
        """

        self.__update()
        return self.last_summary

    def __update(self):
        """
        updates the last_summary and last_string internal attributes
        """
        summary = {
        'shape': self.input_array.shape,
        'size': self.input_array.size,
        'max': round(self.input_array.max(), 3),
        'min': round(self.input_array.min(), 3),
        'mean': round(self.input_array.mean(), 3),
        'dtype': self.input_array.dtype,
        }

        string = "[ARRAY SUMMARY | "  \
                + "shape: {shape} | " \
                + "size: {size} | " \
                + "max: {max} | " \
                + "min: {min} | " \
                + "mean: {mean} | " \
                + "dtype: {dtype}]"

        string = string.format( **summary )

        self.last_summary = summary
        self.last_string = string

        self.update(self.last_summary)


    def __repr__(self):
        return str(self)
