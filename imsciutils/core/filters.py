# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .coordinates import dimensions
import numpy as np

def low_pass(img,cut_off,filter_type='ideal',butterworth_order=1):
    """calculates a lowpass filter for an input image

    Args:
        img(np.ndarray): image to calculate filter for
        cut_off (float): cutoff frequency for this filter. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'
        butterworth_order(float): butterworth order if butterworth filter is
            being used

    Returns:
        filter(np.ndarray) 2D filter
    """
    r,c,b,_ = dimensions(img)
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v)
    low_pass = np.sqrt( (u-r/2)**2 + (v-c/2)**2 )

    if filter_type == 'ideal':
        low_pass[low_pass <= cut_off] = 1
        low_pass[low_pass >= cut_off] = 0

    elif filter_type == 'gaussian':
        xp = -1*(low_pass**2) / (2* cut_off**2)
        low_pass = np.exp( xp )
        low_pass = np.clip(low_pass,0,1)

    elif filter_type == 'butterworth':
        denom = 1.0 + (low_pass / cut_off)**(2 * order)
        low_pass = 1.0 / denom


    return low_pass



def high_pass(img,cut_off,filter_type='ideal',butterworth_order=1):
    """calculates a highpass filter for an input image

    Args:
        img(np.ndarray): image to calculate filter for
        cut_off (float): cutoff frequency for this filter. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'
        butterworth_order(float): butterworth order if butterworth filter is
            being used

    Returns:
        filter(np.ndarray) 2D filter
    """
    return 1 - low_pass(img,cut_off,filter_type='ideal',butterworth_order=1)
