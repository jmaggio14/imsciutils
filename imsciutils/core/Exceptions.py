# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .printout import error as iuerror
import cv2

class CameraReadError(ValueError):
    """Exception raised when the CameraCapture device is unable to
    read the camera
    """
    pass

class InvalidInterpolationType(TypeError):
    """
    Exception for an invalid interpolation Type where it's applicable

    Args:
        interp (cv2.constant): interpolation type
    """
    def __init__(self,interp):
        interp_string = """cv2.INTER_NEAREST --> {}
                        cv2.INTER_LINEAR --> {}
                        cv2.INTER_AREA --> {}
                        cv2.INTER_CUBIC --> {}
                        cv2.INTER_LANCZOS4 --> {}""".format(cv2.INTER_NEAREST,
                                                            cv2.INTER_LINEAR,
                                                            cv2.INTER_AREA,
                                                            cv2.INTER_CUBIC,
                                                            cv2.INTER_LANCZOS4)
        error_string = "'interpolation' ({}) must be one of the following!"\
                                                            .format(interp)
        error_string = error_string + '\n' + interp_string
        iuerror(error_string)
        super(InvalidInterpolationType,self).__init__(error_string)


class InvalidNumpyType(TypeError):
    """
    Exception for an invalid interpolation Type where it's applicable

    Args:
        dtype (np.dtype): numpy datatype
    """
    def __init__(self,dtype):
        error_string = "'dtype' ({}) must be one of the following!"\
                                                            .format(dtype)
        error_string += "\n\t".join(iu.NUMPY_TYPES)
        iuerror(error_string)
        super(InvalidNumpyType,self).__init__(error_string)




class CrackedPipeline(ValueError):
    pass

class BlockRequiresLabels(ValueError):
    pass

class IncompatibleTypes(Exception):
    pass

class InvalidBlockInputData(TypeError):
    def __init__(self,block):
        error_msg = "invalid input to block: {}, must be a list containing ({})".format(
            block.name,
            block.input_shape,
        )
        iuerror(error_msg)
        super(InvalidBlockInputData,self).__init__(error_msg)

class InvalidBlockInputLabels(TypeError):
    def __init__(self,block):
        error_msg = "{}: input labels must a list or NoneType".format(
            block.name,
        )
        iuerror(error_msg)
        super(InvalidBlockInputData,self).__init__(error_msg)

class InvalidProcessStrategy(TypeError):
    def __init__(self,block):
        error_msg = "{}: function 'batch_process' must return a list!".format(
            block.name)
        iuerror(error_msg)
        super(InvalidProcessStrategy,self).__init__(error_msg)

class InvalidLabelStrategy(TypeError):
    def __init__(self,block):
        error_msg = "{}: function 'labels' must return a list or NoneType!".format(
            block.name)
        iuerror(error_msg)
        super(InvalidLabelStrategy,self).__init__(error_msg)

class DataLabelMismatch(TypeError):
    def __init__(self,processed,labels):
        error_msg = "you must have an equal number of processed ({}) and labels ({}). "
        error_msg += "Perhaps the size of your dataset is changing? "
        error_msg += "If so, then you'll have to modify number of labels, "
        error_msg += "look into overloading 'before_process', 'labels', "
        errror_msg += "or 'label' depending on your system".format(
            len(processed),
            len(labels)
            )
        iuerror(error_msg)
        super(DataLabelMismatch,self).__init__(error_msg)
