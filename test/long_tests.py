#!/usr/bin/env python

"""
Adds testing functions in this file, any functions you write will
be automatically run
    -JM
"""
from __future__ import print_function

import imsciutils as iu
VERBOSE = False

@iu.unit_test
def test_multilayer_perceptron():
    import imsciutils as iu

    resizer = iu.Resizer(32,32) #28x28
    features = iu.PretrainedNetwork() # generate features
    pca = iu.PCA(256)
    classifier = iu.MultilayerPerceptron(neurons=512,
                                            validation=.1,
                                            num_hidden=3,
                                            dropout=.35,
                                            learning_rate=.01) # NN classifier
    # there are a lot more parameters you can tweak!

    pipeline = iu.Pipeline([resizer,features,pca,classifier])
    pipeline.rename('test_multilayer_perceptron')
    pipeline.debug()

    # for this example, we'll need to load the standard Mnist handwriting dataset
    # built into `imsciutils`
    mnist = iu.Mnist()
    train_data, train_labels = mnist.get_train()
    test_data, ground_truth = mnist.get_test()

    # train the classifier
    pipeline.train(train_data,train_labels)

    # test the classifier
    predictions = pipeline.process(test_data)

    # print the accuracy
    accuracy = iu.accuracy(predictions,ground_truth)
    print('accuracy is ', accuracy)

    if len(predictions) == len(test_data):
        return True
    return False


@iu.unit_test
def test_linear_svm():
    import imsciutils as iu

    resizer = iu.Resizer(32,32) #28x28
    features = iu.PretrainedNetwork() # generate features
    pca = iu.PCA(256)
    classifier = iu.LinearSvm()

    pipeline = iu.Pipeline([resizer,features,pca,classifier])
    pipeline.rename('test_linear_svm')
    pipeline.debug()

    # for this example, we'll need to load the standard Mnist handwriting dataset
    # built into `imsciutils`
    mnist = iu.Mnist()
    train_data, train_labels = mnist.get_train()
    test_data, ground_truth = mnist.get_test()

    # train the classifier
    pipeline.train(train_data,train_labels)

    # test the classifier
    predictions = pipeline.process(test_data)

    # print the accuracy
    accuracy = iu.accuracy(predictions,ground_truth)
    print('accuracy is ', accuracy)

    if len(predictions) == len(test_data):
        return True
    return False


@iu.unit_test
def test_rbf_svm():
    import imsciutils as iu

    resizer = iu.Resizer(32,32) #28x28
    features = iu.PretrainedNetwork() # generate features
    pca = iu.PCA(256)
    classifier = iu.RbfSvm()

    pipeline = iu.Pipeline([resizer,features,pca,classifier])
    pipeline.rename('test_rbf_svm')
    pipeline.debug()

    # for this example, we'll need to load the standard Mnist handwriting dataset
    # built into `imsciutils`
    mnist = iu.Mnist()
    train_data, train_labels = mnist.get_train()
    test_data, ground_truth = mnist.get_test()

    # train the classifier
    pipeline.train(train_data,train_labels)

    # test the classifier
    predictions = pipeline.process(test_data)

    # print the accuracy
    accuracy = iu.accuracy(predictions,ground_truth)
    print('accuracy is ', accuracy)

    if len(predictions) == len(test_data):
        return True
    return False


@iu.unit_test
def test_poly_svm():
    import imsciutils as iu

    resizer = iu.Resizer(32,32) #28x28
    features = iu.PretrainedNetwork() # generate features
    pca = iu.PCA(256)
    classifier = iu.PolySvm()

    pipeline = iu.Pipeline([resizer,features,pca,classifier])
    pipeline.rename('test_poly_svm')
    pipeline.debug()

    # for this example, we'll need to load the standard Mnist handwriting dataset
    # built into `imsciutils`
    mnist = iu.Mnist()
    train_data, train_labels = mnist.get_train()
    test_data, ground_truth = mnist.get_test()

    # train the classifier
    pipeline.train(train_data,train_labels)

    # test the classifier
    predictions = pipeline.process(test_data)

    # print the accuracy
    accuracy = iu.accuracy(predictions,ground_truth)
    print('accuracy is ', accuracy)

    if len(predictions) == len(test_data):
        return True
    return False

@iu.unit_test
def test_sigmoid_svm():
    import imsciutils as iu

    resizer = iu.Resizer(32,32) #28x28
    features = iu.PretrainedNetwork() # generate features
    pca = iu.PCA(256)
    classifier = iu.SigmoidSvm()

    pipeline = iu.Pipeline([resizer,features,pca,classifier])
    pipeline.rename('test_sigmoid_svm')
    pipeline.debug()

    # for this example, we'll need to load the standard Mnist handwriting dataset
    # built into `imsciutils`
    mnist = iu.Mnist()
    train_data, train_labels = mnist.get_train()
    test_data, ground_truth = mnist.get_test()

    # train the classifier
    pipeline.train(train_data,train_labels)

    # test the classifier
    predictions = pipeline.process(test_data)

    # print the accuracy
    accuracy = iu.accuracy(predictions,ground_truth)
    print('accuracy is ', accuracy)

    if len(predictions) == len(test_data):
        return True
    return False






@iu.unit_test
def test_all_pretrained_networks():
    import imsciutils as iu
    import cv2

    filenames = iu.standard_image_filenames()
    images = [cv2.imread(f,cv2.IMREAD_COLOR) for f in filenames]
    printer = iu.get_printer('test_all_pretrained_networks')

    success = []
    for i,network_name in enumerate(iu.PRETRAINED_NETWORKS):
        try:
            printer.info("testing {}...".format(network_name))
            resizer = iu.Resizer(80,80)
            pretrained = iu.PretrainedNetwork(network_name)

            pipeline = iu.Pipeline([resizer,pretrained])
            pipeline.process(images)

            del pretrained
            del pipeline
            success.append(True)
            printer.info("{}/{} test successful!"\
                            .format(i+1,len(iu.PRETRAINED_NETWORKS)))


        except Exception as e:
            printer.error("failure processing ",network_name)
            success.append(False)

    return all(success)


import six
if six.PY3:
    import queue
else:
    import Queue as queue

Q = queue.Queue()

def prevent_travis_timeout():
    import time
    import sys
    from datetime import datetime

    exit_status = False
    idx = 0
    while True:
        if not idx % 300 or idx == 0:
            print("{} : this message is printed to prevent travis from timing out the test".format(datetime.now()))

        #check queue
        if not Q.empty():
            exit_status = Q.get_nowait()

        if exit_status:
            sys.exit()

        time.sleep(1) # sleep for 5minutes
        idx += 1



def main(verbose=False):
    """
    runs all other function in this file automatically and prints out success
    or failure
    """
    import imsciutils as iu
    import six
    import threading

    travis_timeout = threading.Thread(target=prevent_travis_timeout)
    travis_timeout.start()

    if verbose:
        if six.PY2:
            print('verbose options are not available in python2!')
        else:
            global VERBOSE
            VERBOSE = True
    else:
        iu.disable_all_printers()

    import sys
    unit_tests = [var for var in globals().values() if callable(var)]
    success = []
    for test_func in unit_tests:
        if test_func.__name__ in ['main','prevent_travis_timeout']:
            # skipping main function to avoid recursion loop
            continue
        else:
            success.append( test_func() )

    print("tests completed!")
    print("closing travis timeout thread...")
    Q.put_nowait(True) #tell the travis thread to exit

    # Exit with a 1 code if more than 1 unit test failed
    if not all(success):
        print('not all unit tests passed. add --verbose for more details')
        sys.exit( 1 )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        help='whether or not to print out the arguments passed into functions that use Tester',
                        action='store_true')
    args = parser.parse_args()
    main(args.verbose)






# END
