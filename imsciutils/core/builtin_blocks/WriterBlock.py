# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .. import SimpleBlock
from .. import ArrayType
from ... import util
import cv2
import os




class WriterBlock(SimpleBlock):
    """
    Block that operates as a system that saves single frames to a
    specified output directory.

    this block will not impact dataflow through the pipeline and will return
    whatever images are passed into it

    every new frame passed in will be saved in the following format:
        output_dir/000001base_filename.png
        output_dir/000002base_filename.png
        output_dir/000003base_filename.png
        ...

    Args:
        output_dir (str):
            path to output directory that images will be saved to
        base_filename (str): default is 'image.png'
            filename common among all images, these will be incremented
            numerically with each new image saved
    """
    def __init__(self,
                    output_dir,
                    base_filename='image.png',
                    return_type='filename'):

        assert isinstance(output_dir, str), "'output_dir' must be str"
        assert isinstance(base_filename, str), "'base_filename' must be str"
        assert return_type in ['filename','datum'],\
            "'return_type' must be one of ['filename','datum']"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.base_filename = base_filename
        self.return_type = return_type

        self.image_number = 0
        self.batch_dirs = []
        self.batch_index = 0

        io_map = {ArrayType([None,None,3]):ArrayType([None,None,3]),
                    ArrayType([None,None]):ArrayType([None,None])}

        super(WriterBlock,self).__init__(io_map,
                                            requires_training=False,
                                            requires_labels=False)

    def before_process(self,data,labels):
        """generates output directories for each image in the batch

        Args:
            data(list): list of images to save
            labels(list): labels for each image

        Returns:
            None
        """
        # JM: if integer labels are given, then create different output
        # directories for each new label
        if all(isinstance(lbl,int) for lbl in labels):
            self.batch_dirs = \
                [os.path.join(self.output_dir,str(lbl)) for lbl in labels]
        # JM: otherwise create the same output directory for each image
        else:
            self.batch_dirs = [self.output_dir] * len(data)

        # create output directories if they don't already exist
        uniques = set(self.batch_dirs)
        for out_dir in uniques:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        self.batch_index = 0

    def process(self, datum):
        """writes an image frame to the specificed directory

        Args:
            datum (np.ndarray): frame to be saved to the output_dir

        Returns:
            ret (np.ndarray): datum passed in if return_type == 'datum',
                or the written filename if return_type == 'filename'
        """
        self.image_number += 1
        image_number = util.make_numbered_prefix(self.image_number,6)
        filename = os.path.join(self.batch_dirs[self.batch_index],
                                        image_number + self.base_filename)

        cv2.imwrite(filename, datum)
        self.batch_index += 1
        if self.return_type == 'datum':
            return datum
        else:
            return filename

    def after_process(self):
        self.batch_dirs = []
