# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from ... import util
from .. import BatchBlock
from .. import ArrayType
from random import shuffle
import numpy as np
import zlib
# JM:  keras is imported in the __keras_importer function


class MultilayerPerceptron(BatchBlock):
    """Simple Neural Network Classifier to predict the label of input
    features

    This block utilizes keras in the backend
    This multilayer perceptron supports both integer and one-hot categorical
    labeling during training,

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


    Attributes:
        neurons(int): the number of neurons in each of the
            first and hidden layers
        dropout(float): the fraction of neurons dropped out after each
            layer to mitigate network overfitting
        num_hidden(int): number of layers containing 'neurons'
            fully-connected neurons between the first and last layers
        learning_rate(float): initial learning rate for the SGD optimizer
        decay(float): learning rate decay for the SGD optimizer
        momentum(float): momentum of the SGD optimizer, this is
        batch_size(int): number of datums to processing in each batch,
            larger will improve speed, but increase memory footprint
            default is 128
        label_type(string): the type of labels passed in, must be either
            'categorical' (one-hot) labels or 'integer' labels
            default is integer
        validation(float): the fraction of training data that will be used
            for validating the model. default is 0.0
        model(keras.models.Sequential): the keras model being used in this block
        io_map(IoMap): object that maps inputs to this block to outputs
            subclass of tuple where I/O is stored as:
            ( (input1,output1),(input2,output2)... )
        name(str): unique name for this block
        notes(str): a short description of this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'
        num_epochs(int): the number of epochs to train this model for (the number
            of times the model is trained on the training data). higher usually
            yields better results but linearly increases training time.
            default is 1


    """
    def __init__(self, neurons=512,
                        dropout=0.5,
                        num_hidden=1,
                        learning_rate=0.01,
                        decay=1e-6,
                        momentum=0.9,
                        batch_size=128,
                        label_type='integer',
                        validation=0.0,
                        num_epochs=1,
                        ):

        assert isinstance(neurons, (float, int)),\
                        "neurons must be a float or int"
        assert isinstance(num_hidden, (float, int)),\
                        "num_hidden must be a float or int"
        assert isinstance(dropout, (float, int)),\
                        "dropout must be a float or int"
        assert isinstance(learning_rate, (float, int)),\
                        "learning_rate must be a float or int"
        assert isinstance(decay, (float, int)),\
                        "decay must be a float or int"
        assert isinstance(momentum, (float, int)),\
                        "momentum must be a float or int"
        assert isinstance(batch_size, (float, int)),\
                        "batch_size must be a float or int"
        assert label_type in ['integer', 'categorical'],\
                        "acceptable label_types are ['integer','categorical']"
        assert isinstance(validation, (float, int)),\
                        "validation must be a float or int"
        assert isinstance(num_epochs, (float, int)),\
                        "num_epochs must be an int"

        self.neurons = int(neurons)
        self.dropout = float(dropout)
        self.num_hidden = int(num_hidden)
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)
        self.momentum = float(momentum)
        self.batch_size = int(batch_size)
        self.label_type = label_type
        self.validation = float(validation)
        self.num_epochs = int(num_epochs)

        if self.label_type == 'integer':
            io_map = {ArrayType([1, None]): int}
        else:
            io_map = {ArrayType([1, None]): ArrayType([None], dtypes=np.int32)}

        super(MultilayerPerceptron, self).__init__(io_map,
                                                    requires_training=True,
                                                    requires_labels=True)

    def train(self, data, labels):
        # checking that labels are in the correct format for the type
        if self.label_type == 'integer' and isinstance(labels[0],np.ndarray):
            msg = "'integer' label_type specified, but numpy arrays passed in"\
                    + "-- reseting label_type as 'categorical'!"
            self.printer.warning(msg)
            self.label_type = 'categorical'

        elif self.label_type == 'categorical' and isinstance(labels[0],int):
            msg = "'categorical' label_type specified, but integers passed in"\
                    + "-- reseting label_type as 'integer'!"
            self.printer.warning(msg)
            self.label_type = 'integer'

        from keras.utils import to_categorical
        Sequential, Dense, Dropout, SGD = self.__keras_importer()
        # making the gradient decent optimizer
        sgd = SGD(lr=self.learning_rate,
                    decay=self.decay,
                    momentum=self.momentum,
                    nesterov=True,
                    )

        # calculating the number of unique labels
        if self.label_type == 'integer':
            num_labels = len(np.unique(labels))
        else:
            num_labels = labels.shape[1]

        # stacking the data
        stacked = np.vstack(data)

        # JM: building the model
        self.model = Sequential()
        # building first layer
        self.model.add(Dense(self.neurons,
                                    activation='relu',
                                    input_dim=stacked.shape[1]))
        self.model.add(Dropout(self.dropout))

        # building hidden layers
        for i in range(self.num_hidden):
            self.model.add(Dense(self.neurons, activation='relu'))
            self.model.add(Dropout(self.dropout))

        # adding final layer that scores from 0-1
        self.model.add(Dense(num_labels, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])

        # -------------------- training --------------------
        # JM converting to one-hot labels if integers were passed in
        # this is required for loss='categorical_crossentropy'
        if self.label_type=='integer':
            categorical_labels=to_categorical(labels)
        else:
            categorical_labels=np.array(labels)
        # fitting the model using one-hot labels
        self.model.fit(stacked,
                            categorical_labels,
                            validation_split=self.validation,
                            batch_size=self.batch_size,
                            epochs=self.num_epochs)


    def batch_process(self, data):
        """generates the predicted label given input features"""
        from keras.utils import to_categorical
        # stacking data
        data=np.vstack(data)

        scores=self.model.predict(data,batch_size=self.batch_size)
        # scores here is softmax mapped from 0-1 so the most likely class
        # is simply the max value
        predicted_labels=np.argmax(scores, axis=1)
        # convert to one-hot encoding
        if self.label_type=='categorical':
            predicted_labels=to_categorical(predicted_labels)
            predicted_labels=np.vsplit(predicted_labels.astype(np.int32),
                                                predicted_labels.shape[0])
        else:
            predicted_labels = [int(lbl) for lbl in predicted_labels]

        return predicted_labels


    def __keras_importer(self):
        """imports objects needed from keras, done in separate function
        so keras references remain out of scope for serialization purposes
        (references to multi-threaded or GPU bound memory is unserializable)
        """
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.optimizers import SGD
        return Sequential, Dense, Dropout, SGD


    def prep_for_serialization(self):
        """deletes references to gpu bound or multithreaded memory so the
        block can be serialized, saves pertinent info as instance
        variables so the block can be restored in restore_from_serialization
        """
        # JM: caching model architecture and weights so the model can be
        # restored after serialization
        self.model_architecture = self.model.to_json()
        self.weights = self.model.get_weights()
        # JM: deleting references to gpu bound memory so that this object
        # is now serializable
        delattr(self,'model')

    def restore_from_serialization(self):
        """restores this block from a pickled state
        """
        from keras.models import model_from_json
        # JM: restore model from cached architecture and weights
        self.model = model_from_json(self.model_architecture)
        self.model.set_weights(self.weights)
        # JM: delete cache to reduce memory footprint
        delattr(self,'model_architecture')
        delattr(self,'weights')





# END
