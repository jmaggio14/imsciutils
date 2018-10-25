# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
#
from .. import SimpleBlock, ArrayType
import cv2
import numpy as np



class Otsu(SimpleBlock):
    def __init__(self,min=0,max=255):
        self.min = min
        self.max = max

        io_map = {
                    # ArrayType([None,None]):ArrayType([None,None]),
                    # ArrayType([None,None,3]):ArrayType([None,None,3]),
                    ArrayType([None,None]):ArrayType([None,None]),
                    ArrayType([None,None,3]):ArrayType([None,None,3]),
                    }
        super(Otsu,self).__init__(io_map,
                                    requires_training=False)

    def process(self,datum):
        _,otsu = cv2.threshold(datum.astype(np.uint8),
                                    self.min,
                                    self.max,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return otsu
