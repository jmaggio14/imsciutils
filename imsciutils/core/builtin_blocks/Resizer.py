# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .. import SimpleBlock
from .. import ArrayType
import cv2

class Resizer(SimpleBlock):
    def __init__(self,
                    to_height,
                    to_width,
                    interpolation=cv2.INTER_NEAREST):
        self.to_height = to_height
        self.to_width = to_width
        self.interpolation = interpolation

        io_map = {ArrayType([None,None]):ArrayType([self.to_height,self.to_width]),
                    ArrayType([None,None,3]):ArrayType([self.to_height,self.to_width,3])}

        super(Resizer,self).__init__(io_map,requires_training=False)
        
    def process(self,datum):
        resized = cv2.resize(datum,
                       dsize=(self.to_width,self.to_height),
                       interpolation=self.interpolation)
        return resized
