# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .. import SimpleBlock
from .. import ArrayType
from ..coordinates import dimensions
from ..filters import low_pass,high_pass
import numpy as np

class Highpass(SimpleBlock):
    """Block to apply a highpass filter to a Fourier transform

    Args:
        cut_off (float): cutoff frequency for this filt. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'. default is 'ideal'
        butterworth_order(float): butterworth order if butterworth filter is
            being used. default is 1

    Attributes:
        cut_off (float): cutoff frequency for this filt. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'
        butterworth_order(float): butterworth order if butterworth filter is
            being used

        io_map(IoMap): object that maps inputs to this block to outputs
        name(str): unique name for this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'
    """
    def __init__(self,cut_off,filter_type='ideal',butterworth_order=1):
        self.cut_off = cut_off
        self.filter_type = filter_type
        self.butterworth_order = butterworth_order
        io_map = {
                    ArrayType([None,None]):ArrayType([None,None]),
                    ArrayType([None,None,None]):ArrayType([None,None,None])
                    }
        super(Highpass,self).__init__(io_map, requires_training=False)

    def process(self,datum):
        filt = high_pass(datum,
                    self.cut_off,
                    self.filter_type,
                    self.butterworth_order)

        if datum.ndim == 3:
            filt = np.vstack( (filt,) * filt.shape[2] )

        return filt * datum



class Lowpass(SimpleBlock):
    """Block to apply a lowpass filter to a Fourier transform

    Args:
        cut_off (float): cutoff frequency for this filt. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'. default is 'ideal'
        butterworth_order(float): butterworth order if butterworth filter is
            being used. default is 1

    Attributes:
        cut_off (float): cutoff frequency for this filt. units in #TODO
        filter_type (str): the type of filter to apply, 'ideal','gaussian',
            'butterworth'
        butterworth_order(float): butterworth order if butterworth filter is
            being used

        io_map(IoMap): object that maps inputs to this block to outputs
        name(str): unique name for this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'
    """
    def __init__(self,cut_off,filter_type='ideal',butterworth_order=1):
        self.cut_off = cut_off
        self.filter_type = filter_type
        self.butterworth_order = butterworth_order
        io_map = {
                    ArrayType([None,None]):ArrayType([None,None]),
                    ArrayType([None,None,None]):ArrayType([None,None,None])
                    }
        super(Lowpass,self).__init__(io_map, requires_training=False)

    def process(self,datum):
        filt = low_pass(datum,
                    self.cut_off,
                    self.filter_type,
                    self.butterworth_order)

        if datum.ndim == 3:
            filt = np.dstack( (filt,) * datum.shape[2] )

        return filt * datum
