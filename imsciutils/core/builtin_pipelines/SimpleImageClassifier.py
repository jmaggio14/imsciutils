# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from ..Pipeline import Pipeline
from .. import builtin_blocks as blocks



def SimpleImageClassifier(neurons=512,
                            dropout=.50,
                            num_hidden=2,
                            learning_rate=0.01,
                            decay=1e-6,
                            momentum=0.9,
                            batch_size=128,
                            label_type='integer',
                            validation=0.0,
                            num_epochs=1,
                            pca_components=256,
                            pretrained_network='densenet121',
                            pooling_type='avg',
                            ):
    """returns a simple image classifier pipeline composed of the following
    blocks.
    This pipeline will take in image filenames and return the predicted label

    ImageLoader --> PretrainedNetwork --> PCA --> MultilayerPerceptron

    Args:
        neurons(int): the number of neurons in each of the
            first and hidden layers
        dropout(float): the fraction of neurons dropped out after each
            layer to mitigate network overfitting
        num_hidden(int): number of layers containing 'neurons'
            fully-connected neurons between the first and last layers.
            This is the parameter to tweak to make the network _deeper_
        learning_rate(float): initial learning rate for the SGD optimizer
        decay(float): learning rate decay for the SGD optimizer
        momentum(float): momentum of the SGD optimizer, this affects the
            descent rate and oscillation dampening
        batch_size(int): number of datums to train on in each batch,
            larger will improve speed, but increase memory footprint
            default is 128
        label_type(string): the type of labels passed in, must be either
            'categorical' (one-hot) labels or 'integer' labels
            default is integer
        validation(float): the fraction of training data that will be used
            for validating the model. default is 0.0
        num_epochs(int): the number of epochs to train this model for (the number
            of times the model is trained on the training data). higher usually
            yields better results but linearly increases training time.
            default is 1
        pca_components(int): the number of dimensions to reduce the feature
            vector to. default is 256
        network_name (str): name of network to extract features from
            Default is 'densenet121'
        pooling_type (str): the type of pooling to perform on the
            features, must be one of ['max','avg'].
            Default is 'avg'

    Returns:
        classifier(iu.Pipeline): a pipeline that will process and classify
            input imagery after training

    """
    loader = blocks.ImageLoader()
    features = iu.PretrainedNetwork(network=pretrained_network,
                                    pooling_type=pooling_type)
    pca = iu.PCA(pca_components)
    perceptron = iu.MultilayerPerceptron(neurons=neurons,
                                            dropout=dropout,
                                            num_hidden=num_hidden,
                                            learning_rate=learning_rate,
                                            decay=decay,
                                            momentum=momentum,
                                            batch_size=batch_size,
                                            label_type=label_type,
                                            validation=validation,
                                            num_epochs=num_epochs,
                                            )
    pipeline = Pipeline([loader,features,pca,perceptron],
                            name='SimpleImageClassifier').debug()

    return pipeline
