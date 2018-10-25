# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .. import SimpleBlock
from .. import ArrayType
from ..coordinates import dimensions
import cv2
import numpy as np


class Color2Gray(SimpleBlock):
    """Block to convert color imagery to greyscale imagery

    Args:
        order(str): the channel order of the imagery, must be 'rgb' or 'bgr'
            default is 'rgb'

    Attributes:
        order(str): the channel order for input images
        flag(cv2 constant): opencv flag to determine channel order

        io_map(IoMap): object that maps inputs to this block to outputs
        name(str): unique name for this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'
    """
    def __init__(self, order='rgb'):
        if order == 'rgb':
            self.flag = cv2.COLOR_RGB2GRAY
        elif order == 'bgr':
            self.flag = cv2.COLOR_BGR2GRAY
        else:
            raise ValueError("unknown channel order, must be 'rgb' or 'bgr'")

        self.order = order

        io_map = {
                    ArrayType([None,None],[None,None,3]):
                                        ArrayType([None,None]),
                    # ArrayType([None,None],[None,None,3]):
                    #                     ArrayType([None,None]),
                    # ArrayType([None,None],[None,None,3]):
                    #                     ArrayType([None,None])
                }

        super(Color2Gray,self).__init__(io_map,requires_training=False)

    def process(self,datum):
        """converts color image to grayscale
        converts to grayscale, or does nothing if image is already grayscale

        Args:
            datum(np.ndarray): image to convert

        Returns:
            gray(np.ndarray): grayscale image
        """
        if datum.ndim == 2:
            # Image is already grayscale
            return datum

        gray = cv2.cvtColor(datum.astype(np.uint8), self.flag)
        return gray
