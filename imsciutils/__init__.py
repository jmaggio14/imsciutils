# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import pkg_resources
STANDARD_IMAGE_DIRECTORY = pkg_resources.resource_filename(__name__,
                                                        'data/standard_images')


from .version_info import *
from .core import *
from . import util
from . import ml
from . import io
