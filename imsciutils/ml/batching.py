# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import cv2
from collections import Iterable
import numpy as np

from .. import core


def batches(data_list,batch_size):
    if batch_size > len(data_list):
        error_msg = "batch_size cannot be greater than the length of your dataset"
        core.error(error_msg)
        raise ValueError(error_msg)

    data_length = len(data_list)
    starts = list( range(0,data_length,batch_size) )
    ends = list( range(batch_size,data_length,batch_size) ) + [data_length]

    for start,end in zip(starts,ends):
        batch = data_list[start:end]
        yield batch

def load_and_stack_images(img_list):
    assert isinstance(img_list,Iterable),"img_list must be an iterable type"

    imgs = []
    for img in img_list:
        try:
            loaded = cv2.imread(img,cv2.IMREAD_UNCHANGED)
        except Exception as e:
            core.debug(e)
            core.error( "unable to open img: {}".format(img) )

        r,c,b,_ = core.dimensions(img)
        loaded = loaded.reshape( (1,r,c,b) )
        imgs.append(load)

    return np.vstack(imgs)
