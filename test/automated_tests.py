#!/usr/bin/env python

"""
Adds testing functions in this file, any functions you write will
be automatically run
    -JM
"""
from __future__ import print_function

import imsciutils as iu
VERBOSE = False
# constants.py
@iu.unit_test
def test_constants():
    import imsciutils as iu
    testing_printer = iu.get_printer('test_constants')
    if not 'CV2_INTERPOLATION_TYPES' in dir(iu):
        return False
    if not 'NUMPY_TYPES' in dir(iu):
        return False
    if not 'IMAGE_EXTENSIONS' in dir(iu):
        return False
    if not 'PRETRAINED_NETWORKS' in dir(iu):
        return False

    return True


# coordinates.py
@iu.unit_test
def test_centroid():
    import imsciutils as iu
    testing_printer = iu.get_printer('test_centroid')
    tester = iu.Tester(iu.centroid,verbose=VERBOSE)
    lenna = iu.lenna()
    desired_output = (256,256)
    if not tester.exact_test(desired_output, lenna):
        return False

    return True


@iu.unit_test
def test_frame_size():
    import imsciutils as iu
    testing_printer = iu.get_printer('test_frame_size')
    tester = iu.Tester(iu.frame_size,verbose=VERBOSE)
    lenna = iu.lenna()
    desired_output = tuple( lenna.shape[:2] )
    if not tester.exact_test(desired_output,lenna):
        return False

    return True


@iu.unit_test
def test_dimensions():
    import imsciutils as iu
    testing_printer = iu.get_printer('test_dimensions')
    tester = iu.Tester(iu.dimensions,verbose=VERBOSE)
    lenna = iu.lenna()
    # tuple test
    desired_output = (lenna.shape[0],lenna.shape[1],lenna.shape[2],lenna.dtype)
    if not tester.exact_test(desired_output,lenna):
        return False
    # dimensions test
    desired_output = {
                        'rows':lenna.shape[0],
                        'cols':lenna.shape[1],
                        'bands':lenna.shape[2],
                        'dtype':lenna.dtype,
                        }
    if not tester.exact_test(desired_output,lenna,return_as_dict=True):
        return False

    return True



# img_tools.py
@iu.unit_test
def test_normalize_and_bin():
    import imsciutils as iu
    import numpy as np
    testing_printer = iu.get_printer('test_normalize_and_bin')
    tester = iu.Tester(iu.normalize_and_bin,verbose=VERBOSE)
    lenna = iu.lenna()
    desired_output = np.uint8(lenna.astype(np.float32) / lenna.max() * 255)
    if not tester.exact_test(desired_output,lenna):
        return False

    return True



@iu.unit_test
def test_imageloader_resizer_color2gray_orb_pipeline():
    import imsciutils as iu
    import numpy as np
    testing_printer = iu.get_printer('test_imageloader_resizer_color2gray_orb_pipeline')
    ORB_KEYPOINTS = 10
    # creating all the blocks for the pipeline
    image_loader = iu.ImageLoader()
    resizer = iu.Resizer(to_height=512,to_width=512)
    color2gray = iu.Color2Gray('rgb')
    orb = iu.Orb(n_keypoints=ORB_KEYPOINTS)

    # creating pipeline with all blocks
    pipeline = iu.Pipeline(name='test_imageloader_resizer_color2gray_orb_pipeline',
                            blocks=[image_loader,resizer,color2gray,orb],
                            enable_text_graph=True)
    pipeline.printer.set_log_level('debug')
    iu.set_global_printout_level(0)

    # getting sample data for this system
    standard_image_filenames = iu.standard_image_filenames()
    processed = pipeline.process(standard_image_filenames)

    if all(p.shape == (ORB_KEYPOINTS,32) for p in processed):
        return True
    else:
        testing_printer.info("incorrect shape of outputs")
        return False


@iu.unit_test
def test_imageloader_resizer_color2gray_orb_pipeline():
    import imsciutils as iu
    import numpy as np
    testing_printer = iu.get_printer('test_imageloader_resizer_color2gray_orb_pipeline')
    ORB_KEYPOINTS = 10
    # creating all the blocks for the pipeline
    image_loader = iu.ImageLoader()
    resizer = iu.Resizer(to_height=512,to_width=512)
    color2gray = iu.Color2Gray('rgb')
    orb = iu.Orb(n_keypoints=ORB_KEYPOINTS)

    # creating pipeline with all blocks
    pipeline = iu.Pipeline(name='test_imageloader_resizer_color2gray_orb_pipeline',
                            blocks=[image_loader,resizer,color2gray,orb],
                            enable_text_graph=True).debug()
    pipeline.printer.set_log_level('debug')
    iu.set_global_printout_level(0)

    # getting sample data for this system
    standard_image_filenames = iu.standard_image_filenames()
    processed = pipeline.process(standard_image_filenames)

    if all(p.shape == (ORB_KEYPOINTS,32) for p in processed):
        return True
    else:
        testing_printer.info("incorrect shape of outputs")
        return False


@iu.unit_test
def test_dataset_mnist():
    import imsciutils as iu
    import numpy as np

    TRAINING_LENGTH = 60000
    TESTING_LENGTH = 10000
    NUM_LABELS = 10


    dataset = iu.Mnist()

    train_data,train_labels = dataset.get_train()
    test_data,test_labels = dataset.get_test()

    assert len(train_data) == TRAINING_LENGTH,"train data is incorrect length"
    assert len(train_labels) == TRAINING_LENGTH,"train labels is incorrect length"
    assert len(test_data) == TESTING_LENGTH,"test data is incorrect length"
    assert len(test_labels) == TESTING_LENGTH,"test labels is incorrect length"

    sorted_train_data,sorted_train_labels = dataset.get_sorted_train()
    sorted_test_data,sorted_test_labels = dataset.get_sorted_test()

    assert sorted(sorted_train_labels) == sorted_train_labels
    assert sorted(sorted_test_labels) == sorted_test_labels

    assert len(np.unique(train_labels)) == NUM_LABELS


    return True



@iu.unit_test
def test_dataset_mnist_fashion():
    import imsciutils as iu
    import numpy as np

    TRAINING_LENGTH = 60000
    TESTING_LENGTH = 10000
    NUM_LABELS = 10

    dataset = iu.MnistFashion()

    train_data,train_labels = dataset.get_train()
    test_data,test_labels = dataset.get_test()

    assert len(train_data) == TRAINING_LENGTH,"train data is incorrect length"
    assert len(train_labels) == TRAINING_LENGTH,"train labels is incorrect length"
    assert len(test_data) == TESTING_LENGTH,"test data is incorrect length"
    assert len(test_labels) == TESTING_LENGTH,"test labels is incorrect length"

    sorted_train_data,sorted_train_labels = dataset.get_sorted_train()
    sorted_test_data,sorted_test_labels = dataset.get_sorted_test()

    assert sorted(sorted_train_labels) == sorted_train_labels
    assert sorted(sorted_test_labels) == sorted_test_labels

    assert len(np.unique(train_labels)) == NUM_LABELS


    return True

@iu.unit_test
def test_dataset_cifar10():
    import imsciutils as iu
    import numpy as np

    TRAINING_LENGTH = 50000
    TESTING_LENGTH = 10000
    NUM_LABELS = 10

    dataset = iu.Cifar10()

    train_data,train_labels = dataset.get_train()
    test_data,test_labels = dataset.get_test()

    assert len(train_data) == TRAINING_LENGTH,"train data is incorrect length"
    assert len(train_labels) == TRAINING_LENGTH,"train labels is incorrect length"
    assert len(test_data) == TESTING_LENGTH,"test data is incorrect length"
    assert len(test_labels) == TESTING_LENGTH,"test labels is incorrect length"

    sorted_train_data,sorted_train_labels = dataset.get_sorted_train()
    sorted_test_data,sorted_test_labels = dataset.get_sorted_test()

    assert sorted(sorted_train_labels) == sorted_train_labels
    assert sorted(sorted_test_labels) == sorted_test_labels

    assert len(np.unique(train_labels)) == NUM_LABELS


    return True


@iu.unit_test
def test_dataset_cifar100_fine():
    import imsciutils as iu
    import numpy as np

    TRAINING_LENGTH = 50000
    TESTING_LENGTH = 10000
    NUM_LABELS = 100

    dataset = iu.Cifar100('fine')

    train_data,train_labels = dataset.get_train()
    test_data,test_labels = dataset.get_test()

    assert len(train_data) == TRAINING_LENGTH,"train data is incorrect length"
    assert len(train_labels) == TRAINING_LENGTH,"train labels is incorrect length"
    assert len(test_data) == TESTING_LENGTH,"test data is incorrect length"
    assert len(test_labels) == TESTING_LENGTH,"test labels is incorrect length"

    sorted_train_data,sorted_train_labels = dataset.get_sorted_train()
    sorted_test_data,sorted_test_labels = dataset.get_sorted_test()

    assert sorted(sorted_train_labels) == sorted_train_labels
    assert sorted(sorted_test_labels) == sorted_test_labels

    assert len(np.unique(train_labels)) == NUM_LABELS

    return True

@iu.unit_test
def test_dataset_cifar100_coarse():
    import imsciutils as iu
    import numpy as np

    TRAINING_LENGTH = 50000
    TESTING_LENGTH = 10000
    NUM_LABELS = 20

    dataset = iu.Cifar100('coarse')

    train_data,train_labels = dataset.get_train()
    test_data,test_labels = dataset.get_test()

    assert len(train_data) == TRAINING_LENGTH,"train data is incorrect length"
    assert len(train_labels) == TRAINING_LENGTH,"train labels is incorrect length"
    assert len(test_data) == TESTING_LENGTH,"test data is incorrect length"
    assert len(test_labels) == TESTING_LENGTH,"test labels is incorrect length"

    sorted_train_data,sorted_train_labels = dataset.get_sorted_train()
    sorted_test_data,sorted_test_labels = dataset.get_sorted_test()

    assert sorted(sorted_train_labels) == sorted_train_labels
    assert sorted(sorted_test_labels) == sorted_test_labels

    assert len(np.unique(train_labels)) == NUM_LABELS

    return True


# @iu.unit_test
# def test_thresholding_otsu():
#     import imsciutils as iu
#     import os
#
#     testing_printer = iu.get_printer('otsu_thresholding')
#
#     standard_image_filenames = iu.standard_image_filenames()
#
#     # build blocks for this pipeline
#     loader = iu.ImageLoader()
#     gray2color = iu.Gray2Color()
#     otsu = iu.Otsu()
#     writer = iu.WriterBlock('./output_dir',return_type='filename')
#
#     # pipeline construction
#     pipeline = iu.Pipeline([loader,gray2color,otsu,writer])
#
#     # get filenames of saved thresholded data
#     processed_filenames = pipeline.process(standard_image_filenames)
#
#     for fname in processed_filenames:
#         if not os.path.exists(fname):
#             testing_printer.info("{} was not properly saved".format(fname))
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
