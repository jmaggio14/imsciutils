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


class Gray2Color(SimpleBlock):
    """stacks copies of a 2D array into a 3D image to convert to a grayscale
        color image, this may be required for neural networks which are built
        for RGB data

    Attributes:
        io_map(IoMap): object that maps inputs to this block to outputs
        name(str): unique name for this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'
    """
    def __init__(self):
        io_map = {
                    ArrayType([None,None],[None,None,3]):
                                        ArrayType([None,None,3]),
                    # ArrayType([None,None],[None,None,3]):
                    #                     ArrayType([None,None,3]),
                    # ArrayType([None,None],[None,None,3]):
                    #                     ArrayType([None,None,3])
                }
        super(Gray2Color,self).__init__(io_map,requires_training=False)

    def process(self,datum):
        """converts grayscale to color image
        converts to rgb, or does nothing if image is already rgb

        Args:
            datum(np.ndarray): grayscale image to convert

        Returns:
            rgb(np.ndarray): rgb image
        """
        if datum.ndim == 3:
            # Image is already rgb
            return datum

        rgb = cv2.cvtColor(datum.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return rgb
