# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from ... import util
from .. import BatchBlock
from .. import ArrayType





class CameraBlock(BatchBlock):
    """block to retrieve imagery from a camera

    This block is meant to serve as a entry point for a pipeline by capturing
    images from a UVC-compatible camera. Unlike most blocks in imsciutils,
    this block will have more outputs than inputs. The number of images that should
    be captured is specified be a single element list: [num]

    if mode is set to 'time', then num will represent the duration of time
    in seconds to capture images for

    if mode is set to 'count', then num will represent the number of images
    to capture regardless of time

    N pics to capture (length=1) --> CameraBlock --> N images (length=N)

    Args:
        device(int,str):
            the file path to the camera, or alternatively the camera's
            numerical device id (on linux, this number is at the end of
            the camera's file path eg: "/dev/video0")
            default is 0.
        fourcc(str):
            the codec used to encode images off the camera. Many UVC
            camera device achieve highest frame rates with MJPG
            default is 'MJPG'.
            see: https://docs.microsoft.com/en-us/windows/desktop/medfound/video-fourccs
        mode(str):
            the mode for this block to operate in, either 'count' mode or 'time'
            mode. default is 'count'

    Attributes:
        device(int,str):
            the file path to the camera, or alternatively the camera's
            numerical device id (on linux, this number is at the end of
            the camera's file path eg: "/dev/video0")
        fourcc(str):
            the codec used to encode images off the camera. Many UVC
            camera device achieve highest frame rates with MJPG
            default is 'MJPG'.
            see: https://docs.microsoft.com/en-us/windows/desktop/medfound/video-fourccs
        mode(str):
            the mode for this block to operate in, either 'count' mode or 'time'
            mode. default is 'count'
        
        io_map(IoMap): object that maps inputs to this block to outputs
        name(str): unique name for this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'
    """
    def __init__(self,device=0,fourcc='MJPG',mode='count'):
        #JM: error checking for these values will occur in io.CameraCapture
        self.device = device
        self.fourcc = fourcc

        assert mode in ['count','time'], "mode must set to 'time' or 'count'"
        self.mode = mode

        io_map = {int:ArrayType([None,None],[None,None,3]),
                    float:ArrayType([None,None],[None,None,3]),
                    }


        from ... import io
        self.cap = io.CameraCapture(self.device,self.fourcc)
        super(CameraBlock,self).__init__(io_map, requires_training=False)

    def before_process(self,data,labels=None):
        images = []
        image_labels = []

        if labels is None:
            labels = [None]


        if self.mode == 'count':
            # JM: data in this case should be a 1 element list with the number
            # of images to capture
            num_images = int(data[0])
            lbl = labels[0]
            for i in range(num_images):
                img = self.cap.retrieve()
                images.append(img)
                images_labels.append(lbl)

        elif self.mode == 'time':
            # JM: data in this case should be a 1 element list with the number
            # of seconds to capture frames for
            num_seconds = float(data[0])
            lbl = labels[0]
            # JM: make timer to set a countdown until capture should stop
            t = util.Timer()
            t.countdown = num_seconds
            # JM: retrieve images until countdown hits zero
            while t.countdown:
                img = self.cap.retrieve()
                images.append(img)
                image_labels.append(lbl)

        # set captured images as a instance variable so it can be accessed
        # in batch_process
        self.images = images
        self.image_labels = image_labels


    def batch_process(self,data):
        # JM: return image list created in before_process
        return self.images

    def labels(self,labels):
        # JM: return label list created in before_process
        return self.image_labels


    def after_process(self):
        # JM: reset these values to empty lists to reduce idle memory footprint
        self.images = []
        self.image_labels = []
