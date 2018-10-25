# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import cv2
from collections import Iterable
import numpy as np
from functools import partial

from .. import util
from .. import core


# THIS IS A WIP
@core.experimental()
class FisherVectors(object):
    def __init__(self,n_clusters):
        self.n_clusters = self.__error_check(n_clusters)

        self.gmm = cv2.ml.EM_create()
        self.gmm.setClustersNumber(self.n_clusters)
        self.gmm.setCovarianceType(cv2.ml.EM_COV_MAT_DIAGONAL)

    def __error_check(self,n_clusters):
        """checks the instantation args of this class for errors"""
        if not isinstance(n_clusters,(int,float)):
            error_msg = util.type_error_message(n_clusters,'n_clusters',(int,float))
            core.error(error_msg)
            raise TypeError(error_msg)

        return int(n_clusters)


    def validate_data(self,data):
        """
        Asserts that the data passed in is a 3D array of the proper datatypes
        and shape = (n_imgs, n_descriptors, n_features_per_descriptor)
        for most applications, this is synonomous with (n_imgs, n_keypoints, descriptor_length)


        """
        pass


    def fit(self,data):
        self.__validate_data(data)


    def predict(self, data):
        self.__validate_data(data)


    def fit_and_predict(self, data):
        self.fit(data)
        return self.predict(data)

# @iu.experimental()
# class FisherVectors(object):
#     def __init__(self,n_clusters):
#         self.printer = iu.get_printer('Fisher Vector Extractor')
#         if not isinstance(n_clusters,(float,int)):
#             error_msg = util.type_error_message(n_clusters,'n_clusters',(float,int))
#             self.printer.error(error_msg)
#             raise TypeError(error_msg)
#
#         self.n_clusters = int(n_clusters)
#
#         # building the gaussian mixture model
#         self.gmm = cv2.ml.EM_create()
#         self.gmm.setClustersNumber(self.n_clusters)
#         self.gmm.setCovarianceType(cv2.ml.EM_COV_MAT_DIAGONAL)
#
#
#     def array_to_batches(self,data,batch_indices):
#         self.printer.debug("restoring data to original iterable...")
#         last = 0
#         batches = []
#         for i in batch_indices:
#             batches.append( data[last:i,:] )
#             last = i
#
#         return batches
#
#     def batches_to_array(self,data):
#         try:
#             self.printer.debug("stacking data into numpy array for processing...")
#             data = tuple(data)
#             batch_indices = np.cumsum([len(batch) for batch in data])
#             stacked_data = np.vstack( data )
#             return stacked_data, partial(self.array_to_batches, batch_indices=batch_indices)
#
#         except Exception as e:
#             util.debug(e)
#             error_msg = "unable to properly stack data into numpy array!"
#             self.printer.error(error_msg)
#             raise ValueError(error_msg)
#
#
#     def __validate_data(self,data):
#         """
#         input data must be one of two types:
#         data (np.ndarray): numpy array as (n_samples,n_features)
#         OR
#         data (iterable): iterable of numpy arrays as (batch_size,n_features)
#
#         return:
#             train_data (np.ndarray): (n_samples,n_features)
#             batch_return_function (callable): function to return
#         """
#         if util.is_numpy_array(data):
#             is_batches = False
#             if data.ndim != 2:
#                 error_msg = "'data' must be an array of shape (n_samples,n_features)"
#                 self.printer.error(error_msg)
#                 raise ValueError(error_msg)
#             return data, is_batches
#
#         elif util.is_iterable(data):
#             is_batches = True
#             is_stackable = all( (data[0].shape[1:] == d.shape[1:] and d.ndim == 2) for d in data)
#             if not is_stackable:
#                 error_msg = "'every array in the dataset must have the same shape except for the '0' axid"
#                 self.printer.error(error_msg)
#                 raise ValueError(error_msg)
#             return data, is_batches
#
#         else:
#             error_msg = util.type_error_message(data,"data",(Iterable,np.ndarray))
#             self.printer.error(error_msg)
#             raise TypeError(error_msg)
#
#
#     def fit(self,train_data):
#         train_data, is_batches = self.__validate_data(train_data)
#         if is_batches:
#             train_data, _ = self.batches_to_array(data)
#
#         self.gmm.train(train_data)
#         return self
#
#     def fit_and_predict(self,data):
#         data, is_batches = self.__validate_data(data)
#         self.fit(data)
#         data = self.predict(data)
#         return data
#
#     def predict(self,data):
#         data, is_batches = self.__validate_data(data)
#         if is_batches:
#             fvs = []
#             # for d in data:
#
#
# def main():
#     fv = FisherVectors(5)
#
#
# if __name__ == "__main__":
#     main()
