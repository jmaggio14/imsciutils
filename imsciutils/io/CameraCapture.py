# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import numpy as np
import time
import cv2

from .. import core

class CameraCapture(object):
    """
    object used to talk to pull imagery from UVC camera (webcam)

    Args:
        cam (str,int) = 0:
            the file path to the camera, or alternatively the camera's
            numerical device id (on linux, this number is at the end of
            the camera's file path eg: "/dev/video0")

        fourcc (str) = "MJPG":
             the codec used to encode images off the camera. Many UVC
             camera device achieve highest frame rates with MJPG

    Attributes:
        cap (cv2.VideoCapture): the cv2 camera object
        fourcc (str): the fourcc codec used for this camera
        frame_number (int): the number of frame retrieval attempts

    """

    def __init__(self, cam=0, fourcc="MJPG"):
        assert isinstance(cam, (str, int)), "cam' must be str or int"
        assert isinstance(fourcc, str), "'fourcc' must be str"

        if isinstance(cam, str):
            if "/dev/video" in cam:
                cam = cam.replace("/dev/video", "")

        # openning the camera
        self.cam = int(cam)
        self.open()

        # setting the codec
        self._changeable_settings = {
            "width": cv2.CAP_PROP_FRAME_WIDTH,
            "height": cv2.CAP_PROP_FRAME_HEIGHT,
            "fps": cv2.CAP_PROP_FPS,
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "hue": cv2.CAP_PROP_HUE,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "fourcc": cv2.CAP_PROP_FOURCC}
        self.change_setting('fourcc', fourcc)
        self.fourcc = fourcc

    def open(self):
        self.cap = cv2.VideoCapture(self.cam)
        self.frame_number = 0

    def retrieve(self):
        """
        reads an image from the capture stream, returns a static debug
        frame if it fails to read the frame

        Args:
            None

        Returns:
            np.ndarray: image frame from the Capture Stream
        """
        status = False
        self.frame_number += 1
        self.current_frame_id = str(self.frame_number)

        if self.cap.isOpened():
            status, frame = self.cap.read()

        elif not status or not self.cap.isOpened():
            debug_message = "unable to read frame {0}, is camera connected?"\
                .format(self.current_frame_id)
            raise core.CameraReadError(debug_message)

        return frame

    def metadata(self):
        """
        grabs all metadata from the frame using the metadata properties
        and outputs it in an easy to use dictionary. also adds key
        "capture_time", which is the time.time() at the time the metadata
        is collected
        WARNING - what metadata is available is dependent on what
        camera is attached!

        Args:
            None

        Returns:
            dict: dictionary containing all metadata values
        """
        metadata = {
            "width": self.__get_prop(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self.__get_prop(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self.__get_prop(cv2.CAP_PROP_FPS),
            "contrast": self.__get_prop(cv2.CAP_PROP_CONTRAST),
            "brightness": self.__get_prop(cv2.CAP_PROP_BRIGHTNESS),
            "hue": self.__get_prop(cv2.CAP_PROP_HUE),
            "gain": self.__get_prop(cv2.CAP_PROP_GAIN),
            "exposure": self.__get_prop(cv2.CAP_PROP_EXPOSURE),
            "writer_dims": (self.__get_prop(cv2.CAP_PROP_FRAME_HEIGHT),
                            self.__get_prop(cv2.CAP_PROP_FRAME_WIDTH)),
            "fourcc": self.fourcc,
            "fourcc_val": self.__get_prop(cv2.CAP_PROP_FOURCC),
            "capture_time": time.time(),
            "frame_number": self.current_frame_id
        }
        return metadata

    def change_setting(self, setting, value):
        """changes a setting on the capture object
        acceptable

        Args:
            setting (str): The setting to modify. Must be one of
                [width,height,fps,contrast,brightness,hue,gain,
                exposure,writer_dims,fourcc,fourcc_val,
                capture_time,frame_number]
            value (variable): The value to switch the setting to

        Returns:
            None
        """
        if setting not in self._changeable_settings:
            raise ValueError("settings must be one of {0}"
                    .format(self._changeable_settings.keys()))


        flag = self._changeable_settings[setting]
        if setting == 'fourcc':
            value = cv2.VideoWriter_fourcc(*value)
        ret = self.cap.set(flag, value)
        return ret

    def __get_prop(self, flag):
        """
        gets a camera property
        wrapper for VideoCapture.get function

        Args:
            flag (opencv constant): flag indicating what metadata to get

        Returns:
            the camera property requested
        """

        return self.cap.get(flag)

    def close(self):
        self.cap.release()




def main():
    import imsciutils as iu
    cap = iu.io.CameraCapture()
    viewer = iu.Viewer("Camera Capture Test")
    timer = iu.util.Timer()
    timer.countdown = 30
    while timer.countdown:
        img = cap.retrieve()
        metadata = cap.metadata()
        viewer.view(img)
        print(metadata)


if __name__ == "__main__":
    main()
