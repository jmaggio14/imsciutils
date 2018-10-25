# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import os
import glob
import random
import itertools
import collections
import numpy as np
import imsciutils as iu
from .. import core
import fnmatch
from itertools import islice, chain


def chunk(it,size):
    it = iter(it)
    return list(iter(lambda: list(islice(it, size)), ()))

class DatasetManager(object):
    """object to manage and automatically organize your dataset into training
    and testing chunks. This manager supports cross validation.

    Args:
        k_fold(int): the number of folds to rotate the dataset through,
            default is 10
        extensions(list,tuple,None): the file extensions to filter filenames
            with if using the load_from_directories function. default is None
        recursive(bool): whether or not to recursively sort through directories
            if using the load_from_directories function. default is False
        shuffle_seed(None): seed to shuffle datums with. default is None

    Attributes:
        k_fold(int): the number of folds to rotate the dataset through,
        extensions(list,tuple,None): the file extensions to filter filenames
            with if using the load_from_directories function.
        recursive(bool): whether or not to recursively sort through directories
            if using the load_from_directories function.
        shuffle_seed(None): seed to shuffle datums with.
        fold_index(int): the fold this manager is currently on
        class_names(dict): dictionary to containing the name of the classes,
            key is the integer label, value is the the class name
        data_chunks(deque): deque containing all the chunks for the data
        label_chunks(deque): deque containing all the chunks for the labels
        printer(iu.Printer): printer for this class, registered to
            'DatasetManager'
        remaining_folds(int): number of remaining folds
    """
    def __init__(self,
                    k_folds=10,
                    extensions=None,
                    recursive=False,
                    shuffle_seed=None,
                    ):

        assert isinstance(k_folds,(int,float)),"k_folds must be an integer"
        assert isinstance(extensions,(list,tuple,type(None))),\
            "extensions must be a tuple,list or None"
        assert isinstance(shuffle_seed,(int,float,type(None),np.RandomState)),\
            "k_folds must be an integer"

        if extensions is None:
            extensions = ['']

        self.k_folds = int(k_folds)
        self.extensions = tuple(extensions)
        self.recursive = recursive
        self.shuffle_seed = shuffle_seed

        random.seed(self.shuffle_seed)

        self.fold_index = 1
        self.class_names = {}
        self.data_chunks = collections.deque()
        self.label_chunks = collections.deque()
        self.printer = core.get_printer(self.__class__.__name__)

    def rotate(self):
        """rotate the data chunks so the next dataset fold is available"""
        self.fold_index += 1
        self.data_chunks.rotate()
        self.label_chunks.rotate()
        if self.fold_index > self.k_folds:
            warning_msg = "data has been rotated more than the number of folds"
            self.printer.warning(warning_msg)

        return self

    def get_all(self):
        """get the all the data for this fold

        Args:
            None

        Returns:
            data(list): list of all data
            labels(list): list of all labels
        """
        train_x,train_y = self.get_train()
        test_x,test_y = self.get_test()
        return (train_x+test_x), (train_y+test_y)

    def get_train(self):
        """get the training set for this fold

        Args:
            None

        Returns:
            train_data(list): list of training filenames
            train_labels(list): list of training labels
        """
        train_data = [self.data_chunks[i] for i in range(self.k_fold-1)]
        train_labels = [self.labels_chunks[i] for i in range(self.k_fold-1)]

        return list(chain(*train_data)), list(chain(*train_labels))


    def get_test(self):
        """get the testing set for this fold

        Args:
            None

        Returns:
            test_data(list): list of testing filenames
            test_labels(list): list of testing labels
        """
        test_data = self.data_chunks[-1]
        test_labels = self.label_chunks[-1]

        return test_data, test_labels

    def load_from_arrays(self,*arrays,**class_names):
        """load a list of class arrays and apply labels

        Args:
            *arrays: unpacked list of data, each array
                must be for a different class so it can be labeled properly
            class_names(list,tuple): keyword-only argument to specify the name
                of each class

        Return:
            self
        """
        # JM: this is stupid hack to get keyword only arguments in python2
        if len(class_names) > 1:
            raise TypeError("only one keyword argument 'class_names' can be specified")
        if len(class_names) >= 1 and ('class_names' not in class_names):
            raise TypeError("only one keyword argument 'dtypes' can be specified")

        if 'class_names' in class_names:
            assert len(class_names['class_names']) == len(arrays),\
                "you must provide a class name for each array"
            class_names = [str(c) for c in class_names['class_names']]
        else:
            class_names = ['class{}'.format(i) for i in range(len(arrays))]


        # ----- code begins -----
        self.class_names = {}
        data = []
        labels = []
        # load each directory and adding it to the global labels
        for lbl,data_arrays in enumerate(arrays):
            data.extend( data_arrays )
            labels.extend( [lbl] * len(data_arrays) )
            self.class_names[lbl] = class_names[lbl]

        # JM shuffling filenames and labels together
        combined = sorted(zip(data, labels))
        random.shuffle(combined)
        data[:], labels[:] = zip(*combined)

        # chunking all data
        self.data_chunks = collections.deque( chunk(data,k_folds) )
        self.label_chunks = collections.deque( chunk(labels,k_folds) )

        # warning that the fold index has been reset
        if self.fold_index > 1:
            warning_msg = "fold index has been reset because new data was added"
            self.printer.warning(warning_msg)
            self.fold_index = 1

        return self

    def load_from_directories(self,*directories):
        """load a list of class directories and apply labels

        Args:
            *directories: unpacked list of data directories, each directory
                must be for a different class so it can be labeled properly

        Return:
            self
        """
        self.class_names = {}
        filenames = []
        labels = []
        # load each directory and adding it to the global labels
        for lbl,directory in enumerate(directories):
            dir_filenames,dir_labels = self._load_directory(directory,lbl)
            filenames.extend( dir_filenames )
            labels.extend( dir_labels )
            self.class_names[lbl] = os.path.basename(directory)

        # JM shuffling filenames and labels together
        combined = list(zip(filenames, labels))
        random.shuffle(combined)
        filenames[:], labels[:] = zip(*combined)

        # chunking all data
        self.data_chunks = collections.deque( chunk(filenames,k_folds) )
        self.label_chunks = collections.deque( chunk(labels,k_folds) )

        # warning that the fold index has been reset
        if self.fold_index:
            warning_msg = "fold index has been reset because new data was added"
            self.printer.warning(warning_msg)
            self.fold_index = 1

        return self



    def _load_directory(self,directory,label):
        """list all filenames in the directory that match the specified
        extensions.

        Args:
            directory(str): the directory which contains the desired data
            label(int): the label for all the files in this directory

        Returns:
            filenames(list): list of filenames in this directory with
                filtered with the desired extensions
            labels(list): list of labels associated with the files in this
                directory

        Raises:
            AssertionError: if directory is invalid
        """
        assert os.path.isdir(directory),\
            "'{}' must be a valid directory".format(directory)


        filenames = []
        if self.recursive:
            # use os walk and fnmatch instead of glob for python2 compatability
            for ext in self.extensions:
                for root, dirnames, filenames in os.walk(directory):
                    for filename in fnmatch.filter(filenames, '*' + ext):
                        matches.append(os.path.join(root, filename))
        else:
            for ext in self.extensions:
                ext = ext.replace('*','')
                glob_path = os.path.join(directory,'*' + ext)
                filenames.extend( glob.glob(glob_path) )

        return filenames,[label]*len(filenames)

    def get_class_names(self,labels):
        """retrieves names of the classes based on the labels"""
        class_names = [self.class_names[lbl] for lbl in labels]

    def __str__(self):
        return "DatasetManager ({}/{} folds)".format(self.fold_index,
                                                        self.k_folds)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        for i in range(self.remaining_folds):
            train_data,train_labels = self.get_train()
            test_data,test_labels = self.get_test()
            yield train_data,train_labels,test_data,test_labels

    @property
    def remaining_folds(self):
        return (self.k_folds - self.fold_index)
