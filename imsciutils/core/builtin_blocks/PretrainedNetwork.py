
# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from importlib import import_module
import numpy as np
from .. import BatchBlock
from .. import ArrayType
from ..coordinates import dimensions
import cv2

SUBMODULES = {'xception': 'keras.applications.xception',
                'vgg16': 'keras.applications.vgg16',
                'vgg19': 'keras.applications.vgg19',
                'resnet50': 'keras.applications.resnet50',
                'inception_v3': 'keras.applications.inception_v3',
                'inception_resnet_v2': 'keras.applications.inception_resnet_v2',
                'mobilenet': 'keras.applications.mobilenet',
                'densenet121': 'keras.applications.densenet',
                'densenet169': 'keras.applications.densenet',
                'densenet201': 'keras.applications.densenet',
                }
FUNCTION_NAMES = {'xception': 'Xception',
                    'vgg16': 'VGG16',
                    'vgg19': 'VGG19',
                    'resnet50': 'ResNet50',
                    'inception_v3': 'InceptionV3',
                    'inception_resnet_v2': 'InceptionResNetV2',
                    'mobilenet': 'MobileNet',
                    'densenet121': 'DenseNet121',
                    'densenet169': 'DenseNet169',
                    'densenet201': 'DenseNet201',
                    }

MINIMUM_SIZES = {'xception': 71,
                    'vgg16': 32,
                    'vgg19': 32,
                    'resnet50': 32,
                    'inception_v3': 75,
                    'inception_resnet_v2': 75,
                    'mobilenet': 32,
                    'densenet121': 32,
                    'densenet169': 32,
                    'densenet201': 32,
                    }

class PretrainedNetwork(BatchBlock):
    """Block to extract features from pretrained neural networks

    pretrained networks are trained on imagenet with pooling applied.

    if grayscale images are passed into this network, they will be converted
    to RGB (this is required for features to be extracted from a network).

    This class utilizes keras to automatically leverage
    hardware resources and retrieve pretrained networks.
    available networks are:
        - xception
        - vgg16
        - vgg19
        - resnet50
        - inception_v3
        - inception_resnet_v2
        - mobilenet
        - densenet121
        - densenet169
        - densenet201
        - nasnetlarge
        - nasnetmobile
        - mobilenetv2

    see: https://keras.io/applications/ for more details
    kwargs for network instantiation are:
                                        include_top=False,
                                        weights='imagenet',
                                        pooling=self.pooling_type


    Args:
        network_name (str): name of network to extract features from
            Default is 'densenet121'
        pooling_type (str): the type of pooling to perform on the
            features, must be one of ['max','avg'].
            Default is 'avg'

    Attributes:
        network_name (str): name of network to extract features from
            Default is 'densenet121'
        pooling_type (str): the type of pooling to perform on the
            features, must be one of ['max','avg'].
            Default is 'avg'
        model_fn(callable): function to generate features on an image stack
        preprocess_fn(callable): function to preprocess an image stack

        io_map(IoMap): object that maps inputs to this block to outputs
        name(str): unique name for this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'


    """
    def __init__(self,network='densenet121',pooling_type='avg'):
        self.network = network
        self.pooling_type = pooling_type

        # building the keras network
        self.model_fn, self.preprocess_fn, self.min_input_size, output_shape\
            = self._keras_importer(network,pooling_type)

        io_map = {ArrayType([None,None],[None,None,3]):
                                ArrayType([1,output_shape])}
        name = "Pretrained" + self.network[0].upper() + self.network[1:]
        super(PretrainedNetwork,self).__init__(io_map,
                                                name=name,
                                                requires_training=False)

    def batch_process(self,batch_data,batch_labels=None):
        # verify that all images are the same size
        if not all(batch_data[0].shape == d.shape for d in batch_data):
            error_msg = "all input images must be the same shape!"
            self.printer.error(error_msg)
            raise ValueError(error_msg)

        # reshape all images to a 4D array for keras standard
        r,c,b,_ = dimensions( batch_data[0] )
        if (r < self.min_input_size[0]) or (c < self.min_input_size[1]):
            error_msg = "minimum acceptable image size is {}"\
                .format(self.min_input_size)
            self.printer.error(error_msg)
            raise ValueError(error_msg)

        if b == 1:
            batch_data = [cv2.cvtColor(im,cv2.COLOR_GRAY2RGB) for im in batch_data]

        batch_data = [img.reshape(1,r,c,3) for img in batch_data]

        # stack all the images so all data can be processed at once
        img_stack = np.vstack(batch_data)
        img_stack = self.preprocess_fn(img_stack)
        feature_stack = self.model_fn(img_stack)
        features = np.vsplit(feature_stack,feature_stack.shape[0])
        return features


    def _keras_importer(self, network_name, pooling_type):
        """
        Retrieves the feature extraction algorithm and preprocess_fns
        from keras, only importing the desired model specified by the
        network name.

        Args:
            network_name (str):
                name of the network being used for feature extraction
            pooling_type (str):
                type of pooling you want to use ('avg' or 'max')

        Returns:
            model_fn (callable):
                function that extracts features from the NN
            preprocess_fn (callable):
                function that preprocesses the image for the network
            kerasimage (module):
                pointer to keras.image
        """
        # checking to make sure network_name is valid
        if network_name not in SUBMODULES:
            error_string = "unknown network '{network_name}', \
                            network_name must be one of {network_list}"\
                                .format(network_name=network_name,
                                        network_list=SUBMODULES.keys())
            raise ValueError(error_string)

        # checking that poolying type is valid
        assert pooling_type in ['avg','max'],\
            "'pooling_type' must be one of the following strings ['avg','max']"

        # importing the proper keras model_fn and preprocess_fn
        submodule = import_module(SUBMODULES[network_name])
        model_constructor = getattr(submodule,
                                    FUNCTION_NAMES[network_name])
        model = model_constructor(include_top=False,
                                     weights='imagenet',
                                     pooling=pooling_type)
        model_fn = model.predict
        output_shape = model.output_shape[1]
        min_input_size = MINIMUM_SIZES[network_name],MINIMUM_SIZES[network_name]

        preprocess_fn = getattr(submodule, 'preprocess_input')

        # set the keras backend to use channels_last
        from keras import backend as K
        K.set_image_data_format('channels_last')

        return model_fn, preprocess_fn, min_input_size, output_shape
