# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import numpy as np
from itertools import islice, chain
import scipy.stats

def accuracy(predicted,ground_truth):
    """calculates accuracy given ground truth and predicted labels"""
    num_correct = np.sum(np.asarray(predicted) == np.asarray(ground_truth))
    return float(num_correct) / len(predicted)

def confidence_99(data):
    """returns the 99% confidence mean and deviation for the given
        distribution

    Args:
        data(array-like): data to find the confidence interval for,
            in machine learning applications, this is usually accuracy
            for K-fold cross validation

    Returns:
        mean(float): the mean for this distributions
        h(float): +/- deviation for this distribution

    Example:
        >>> # get 99% confidence interval
        >>> confidence_99(accuracies)
    """
    return confidence(data,.99)

def confidence_95(data):
    """returns the 95% confidence mean and deviation for the given
        distribution

    Args:
        data(array-like): data to find the confidence interval for,
            in machine learning applications, this is usually accuracy
            for K-fold cross validation

    Returns:
        mean(float): the mean for this distributions
        h(float): +/- deviation for this distribution

    Example:
        >>> # get 95% confidence interval
        >>> confidence_95(accuracies)
    """
    return confidence(data,.95)

def confidence_90(data):
    """returns the 90% confidence mean and deviation for the given
        distribution

    Args:
        data(array-like): data to find the confidence interval for,
            in machine learning applications, this is usually accuracy
            for K-fold cross validation

    Returns:
        mean(float): the mean for this distributions
        h(float): +/- deviation for this distribution

    Example:
        >>> # get 90% confidence interval
        >>> confidence_90(accuracies)
    """
    return confidence(data,.90)


def confidence(data, confidence=0.95):
    """returns the confidence mean and deviation for the given
        confidence interval

    Args:
        data(array-like): data to find the confidence interval for,
            in machine learning applications, this is usually accuracy
            for K-fold cross validation
        confidence(float): confidence interval between 0-1, to find
            the desired mean and deviation for

    Returns:
        mean(float): the mean for this distributions
        h(float): +/- deviation for this distribution

    Example:
        >>> # get 95% confidence interval
        >>> confidence(accuracies,.95)
    """
    data = np.asarray(data,dtype=np.float32)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, len(data)-1)
    return m, h



def batch(data_list, batch_size):
    """chunks a list into multiple batch_size chunks, the last batch will
    be truncated if the data_list length isn't a multiple of batch_size
    """
    data_list = iter(data_list)
    return list(iter( lambda: list(islice(data_list, size)), ()) )

def batches_to_list(batches):
    """turns nested iterables into a single list"""
    return list( chain(*batches) )
