# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell

# BaseBlock.py
from .BaseBlock import quick_block
from .BaseBlock import ArrayType
from .BaseBlock import IoMap
from .BaseBlock import BaseBlock
from .BaseBlock import SimpleBlock
from .BaseBlock import BatchBlock

# builtin_blocks/
from .builtin_blocks import *

# builtin_pipelines/
from .builtin_pipelines import *

# constants.py
from .constants import *

# coordinates.py
from .coordinates import centroid
from .coordinates import frame_size
from .coordinates import dimensions

# datasets.py
from .datasets import Mnist
from .datasets import MnistFashion
from .datasets import Cifar10
from .datasets import Cifar100

# debug.py
from .debug import debug

# development_decorators.py
from .development_decorators import deprecated
from .development_decorators import experimental
from .development_decorators import human_test
from .development_decorators import print_args
from .development_decorators import unit_test

# error_checking.py
from .error_checking import interpolation_type_check
from .error_checking import dtype_type_check
from .error_checking import is_numpy_array
from .error_checking import is_iterable
from .error_checking import type_error_message

# filters.py
from .filters import low_pass
from .filters import high_pass

# Exceptions.py
from .Exceptions import CameraReadError
from .Exceptions import InvalidInterpolationType
from .Exceptions import InvalidNumpyType
from .Exceptions import CrackedPipeline
from .Exceptions import BlockRequiresLabels
from .Exceptions import IncompatibleTypes
from .Exceptions import InvalidBlockInputData
from .Exceptions import InvalidBlockInputLabels
from .Exceptions import InvalidProcessStrategy
from .Exceptions import InvalidLabelStrategy
from .Exceptions import DataLabelMismatch

# ml_tools.py
from .ml_tools import accuracy
from .ml_tools import confidence_90
from .ml_tools import confidence_95
from .ml_tools import confidence_99
from .ml_tools import confidence
from .ml_tools import batch
from .ml_tools import batches_to_list

# Pipeline.py
from .Pipeline import Pipeline
from .Pipeline import restore_from_file

# Printer.py
from .Printer import get_printer
from .Printer import disable_all_printers
from .Printer import whitelist_printer
from .Printer import blacklist_printer
from .Printer import reset_printer_lists
from .Printer import Printer
from .Printer import disable_printout_colors
from .Printer import enable_printout_colors
from .Printer import get_active_printers
from .Printer import set_global_printout_level

# printout.py
from .printout import get_default_printer
from .printout import debug
from .printout import info
from .printout import warning
from .printout import error
from .printout import critical
from .printout import comment


# standard_image.py
from .standard_image import list_standard_images
from .standard_image import standard_image_filenames
from .standard_image import standard_image_gen
from .standard_image import standard_images
from .standard_image import get_standard_image
from .standard_image import standard_image_input

# ND 9/7/18 - dynamically load each of the previously created convience funcs
# into the current namespace
from .standard_image import STANDARD_IMAGES
from .standard_image import funcs
import sys

curr_module = sys.modules[__name__]
for img_name in STANDARD_IMAGES.keys():
	setattr(curr_module, img_name, getattr(funcs, img_name))

# ND 9/7/18 - delete these so that the imsciutils namespace is not polluted
del sys, curr_module, funcs, STANDARD_IMAGES

# Viewer.py
from .Viewer import Viewer

# img_tools.py
from .img_tools import normalize_and_bin
from .img_tools import quick_image_view
from .img_tools import number_image



# Tester.py
from .Tester import Tester

# Summarizer.py
from .Summarizer import Summarizer
