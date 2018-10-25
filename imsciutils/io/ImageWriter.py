# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import cv2
from .. import core


@core.experimental()
class ImageWriter(object):
    """
    Class that operates as a system that saves single frames to a
    specified output directory.

    every new frame passed in will be saved in the following format:
        output_dir/000001example_filename.png
        output_dir/000002example_filename.png
        output_dir/000003example_filename.png
        ...

    automatic resizing is also available

    Args:
        output_dir (str):
            path to output directory that images will be saved to
        base_filename (str): default is 'image.png'
            filename common among all images, these will be incremented
            numerically with each new image saved
        size (tuple,None): Default is None
            size of the image if forced resizing is desired, or
            None if raw write is desired
        interpolation (cv2 interpolation type):
            Default is cv2.INTER_NEAREST
            interpolation method to be used if resizing is desired

    """

    def __init__(self, output_dir,
                         base_filename="image.png",
                         size=None,
                         interpolation=cv2.INTER_NEAREST):

        assert isinstance(base_filename, str), "'base_filename' must be str"
        imsciutils.util.interpolation_type_check(interpolation)

        self.base_filename = base_filename
        self.size = size
        self.interpolation = interpolation
        self.image_number = 0

    def write(self, frame):
        """
        writes an image frame to the specificed directory, forces
        resizing if specified when the class is instantiated

        Args:
            frame (np.ndarray): frame to be saved to the output_dir

        Returns:
            None
        """
        self.image_number += 1
        image_number = imsciutils.util.make_numbered_prefix(self.image_number,6)
        out_filename = os.path.join(self.output_dir,
                                    image_number + self.base_filename)

        if not isinstance(self.size, type(None)):
            frame = cv2.resize(frame,
                               dsize=(self.size[1], self.size[0]),
                               interpolation=self.interpolation)

        cv2.imwrite(filename, frame)
