# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import numpy as np

class DatasetRetrieval(object):
    def get_sorted_train(self):
        """retrieves data and labels for train set sorted by label"""

        indexes = list(range( len(self.y_train) ))
        indexes.sort(key=self.y_train.__getitem__)
        sorted_data = list( map(self.x_train.__getitem__, indexes) )
        sorted_labels = list( map(self.y_train.__getitem__, indexes) )

        return sorted_data,sorted_labels

    def get_sorted_test(self):
        """retrieves data and labels for test set sorted by label"""

        indexes = list(range( len(self.y_test) ))
        indexes.sort(key=self.y_test.__getitem__)
        sorted_data = list( map(self.x_test.__getitem__, indexes) )
        sorted_labels = list( map(self.y_test.__getitem__, indexes) )

        return sorted_data,sorted_labels


class Mnist(DatasetRetrieval):
    """
    Object to load the MNIST numbers dataset in a pipeline compatible format

    Attributes:
        x_train(list): 60,000 monochromatic 28x28 images
        y_train(list): corresponding integer labels for the data
        x_test(list): 10,000 monochromatic 28x28 images
        y_test(list): corresponding integer labels for the data
    """
    def __init__(self):
        from keras.datasets import mnist
        (x_train,y_train), (x_test,y_test) = mnist.load_data()

        self.x_train = [np.squeeze(x_train[i,:,:]) for i in range(x_train.shape[0])]
        self.y_train = [int(i) for i in y_train]

        self.x_test = [np.squeeze(x_test[i,:,:]) for i in range(x_test.shape[0])]
        self.y_test = [int(i) for i in y_test]

    def get_train(self):
        """
        retrieves the mnist numbers train dataset using keras

        Args:
            None
        Returns:
            x_train(list): 60,000 monochromatic 28x28 images
            y_train(list): corresponding integer labels for the data
        """
        return self.x_train,self.y_train

    def get_test(self):
        """
        retrieves the mnist numbers test dataset using keras

        Args:
            None
        Returns:
            x_test(list): 10,000 monochromatic 28x28 images
            y_test(list): corresponding integer labels for the data
        """
        return self.x_test,self.y_test


class MnistFashion(DatasetRetrieval):
    """
    Object to load the MNIST fashion dataset in a pipeline compatible format

    Attributes:
        x_train(list): 60,000 monochromatic 28x28 images
        y_train(list): corresponding integer labels for the data
        x_test(list): 10,000 monochromatic 28x28 images
        y_test(list): corresponding integer labels for the data
    """
    def __init__(self):
        from keras.datasets import fashion_mnist
        (x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

        self.x_train = [np.squeeze(x_train[i,:,:]) for i in range(x_train.shape[0])]
        self.y_train = [int(i) for i in y_train]

        self.x_test = [np.squeeze(x_test[i,:,:]) for i in range(x_test.shape[0])]
        self.y_test = [int(i) for i in y_test]

    def get_train(self):
        """
        retrieves the mnist numbers train dataset using keras

        Args:
            None
        Returns:
            x_train(list): 60,000 monochromatic 28x28 images
            y_train(list): corresponding integer labels for the data
        """
        return self.x_train,self.y_train

    def get_test(self):
        """
        retrieves the mnist numbers test dataset using keras

        Args:
            None
        Returns:
            x_test(list): 10,000 monochromatic 28x28 images
            y_test(list): corresponding integer labels for the data
        """
        return self.x_test,self.y_test


class Cifar10(DatasetRetrieval):
    """
    Object to load the cifar10 dataset in a pipeline compatible format

    Attributes:
        x_train(list): 50,000 color 32,32,3 images
        y_train(list): corresponding integer labels for the data
        x_test(list): 10,000 color 32,32,3 images
        y_test(list): corresponding integer labels for the data

    """
    def __init__(self):
        from keras.datasets import cifar10
        from keras import backend as K
        K.set_image_data_format('channels_last')
        (x_train,y_train), (x_test,y_test) = cifar10.load_data()

        self.x_train = [np.squeeze(x_train[i,:,:,:]) for i in range(x_train.shape[0])]
        self.y_train = [int(i) for i in y_train]

        self.x_test = [np.squeeze(x_test[i,:,:,:]) for i in range(x_test.shape[0])]
        self.y_test = [int(i) for i in y_test]

    def get_train(self):
        """
        retrieves the cifar 10 labels train dataset using keras

        Args:
            None
        Returns:
            x_train(list): 50,000 color 32,32,3 images
            y_train(list): corresponding integer labels for the data
        """
        return self.x_train,self.y_train

    def get_test(self):
        """
        retrieves the cifar 10 labels test dataset using keras

        Args:
            None
        Returns:
            x_test(list): 10,000 color 32,32,3 images
            y_test(list): corresponding integer labels for the data
        """
        return self.x_test,self.y_test


class Cifar100(DatasetRetrieval):
    """
    Object to load the cifar100 dataset in a pipeline compatible format

    Args:
        label_mode(string): 'fine' for individual labels (100 unique),
            'coarse' for superclass labels (20 unique)

    Attributes:
        x_train(list): 50,000 color 32,32,3 images
        y_train(list): corresponding integer labels for the data
        x_test(list): 10,000 color 32,32,3 images
        y_test(list): corresponding integer labels for the data

    """
    def __init__(self,label_mode='fine'):
        from keras.datasets import cifar100
        from keras import backend as K
        K.set_image_data_format('channels_last')
        (x_train,y_train), (x_test,y_test) = cifar100.load_data(label_mode)

        self.x_train = [np.squeeze(x_train[i,:,:,:]) for i in range(x_train.shape[0])]
        self.y_train = [int(i) for i in y_train]

        self.x_test = [np.squeeze(x_test[i,:,:,:]) for i in range(x_test.shape[0])]
        self.y_test = [int(i) for i in y_test]

    def get_train(self):
        """
        retrieves the cifar 100 labels train dataset using keras

        Args:
            None
        Returns:
            x_train(list): 50,000 color 32,32,3 images
            y_train(list): corresponding integer labels for the data
        """
        return self.x_train,self.y_train

    def get_test(self):
        """
        retrieves the cifar 100 labels test dataset using keras

        Args:
            None
        Returns:
            x_test(list): 10,000 color 32,32,3 images
            y_test(list): corresponding integer labels for the data
        """
        return self.x_test,self.y_test



# END
