# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import cv2
import numpy as np
from datetime import datetime

import imsciutils as iu
from .. import util
from .img_tools import number_image
from .error_checking import interpolation_type_check

class Viewer(object):
    """
    Class to simplify displaying images using opencv windows. Also has
    functionality to resize images automatically if desired

    On some systems (or opencv builds), openCV requires a waitkey to be
    called before displaying the image. This functionality is built in
    the view function with the argument 'force_waitkey'


    Args:
        window_name (str): the name of the window
        size (2 element tuple, None): (height, width) to resize image to
        interpolation (cv2 constant): cv2 interpolation flag for resizing
        enable_frame_counter (bool): whether or not to enable frame counter

    Attributes:
        window_name: name of the cv2 window
        size: size the images are being resized to
        interpolation: type of interpolation being usef for resizing
        enable_frame_counter: whether or not the frame counter is enabled
        frame_counter: index of the frame currently displayed

    """

    def __init__(self,
                 window_name=None,
                 size=None,
                 FFT=False,
                 normalize=False,
                 interpolation=cv2.INTER_NEAREST,
                 enable_frame_counter=False):

        interpolation_type_check(interpolation)
        if window_name is None:
            window_name = datetime.now()

        self.window_name = str(window_name)
        self.size = size
        self._FFT = FFT
        self._normalize = normalize
        self.interpolation = interpolation
        self._enable_frame_counter = enable_frame_counter
        self.open()
        self.frame_counter = 1

    def open(self):
        """
        opens the image viewer, automatically called in __init__

        Args:
            None

        Returns:
            None
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        return self

    def view(self, frame, force_waitkey=True):
        """
        displays the frame passed into it, reopens itself if it

        Args:
            frame (np.ndarray): image to be displayed

        Returns:
            force_waitkey (int) = 1:
                if greater than zero, then call a waitkey for the
                duration of the time given. this is required on some
                systems to display the image properly.
        """
        assert isinstance(frame, np.ndarray), "'frame' must be a numpy array"
        assert frame.ndim in [2, 3], "'frame' must be a 2D or 3D array"
        assert isinstance(force_waitkey, (int, float)),\
            "'force_waitkey' must be an integer"

        # normalize image band by band (useful for FFT viewing)
        if self._FFT:
    
            if frame.ndim == 2:

                frame = 20*np.log(np.abs(frame))

            if frame.ndim == 3:

                for b in range(0, 3):

                    frame[:,:,b] = 20*np.log(np.abs(frame[:,:,b]))

        if self._normalize:

            frame = util.normalize.norm_dtype(frame)

        # cast frame dtype to uint8 for display
        frame = frame.astype(np.uint8)

        if isinstance(self.size, (tuple, list)):
            frame = cv2.resize(frame,
                               dsize=(self.size[1], self.size[0]),
                               interpolation=self.interpolation)

        # add a frame counter to an image thin
        if self._enable_frame_counter:
            frame = number_image(frame,self.frame_counter)
        
        # displaying the image
        cv2.imshow(self.window_name, frame)
        
        if force_waitkey:
            cv2.waitKey(force_waitkey)

        self.frame_counter += 1

    def enable_frame_counter(self):
        """enable the frame counter"""
        self._enable_frame_counter = True

    def disable_frame_counter(self):
        """disable the frame counter"""
        self._enable_frame_counter = False

    def close(self):
        """
        closes the image viewing window

        Args:
            None

        Returns:
            None
        """
        cv2.destroyWindow(self.window_name)
        return self

    def __del__(self):
        self.close()


def main():
    import imsciutils as iu
    import time
    img_gen = iu.standard_image_gen()

    v = iu.ImageViewer('std image test',
                        size=(512,512),
                        enable_frame_counter=True)
    for img,name in zip(img_gen,iu.list_standard_images()):
        iu.info("displaying ",name)
        v.view(img)
        time.sleep(1)


if __name__ == "__main__":
    main()
