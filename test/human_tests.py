import imsciutils as iu

@iu.human_test
def test_quick_image_view():
    """THIS TEST REQUIRES A HUMAN TO VIEW THE IMAGE"""
    import imsciutils as iu
    testing_printer = iu.get_printer('test_quick_image_view')

    pig = iu.pig()
    iu.quick_image_view(pig,title='quick_image_view test')


@iu.human_test
def test_number_image():
    import imsciutils as iu
    testing_printer = iu.get_printer('test_number_image')

    pig = iu.pig()
    linear = iu.linear()

    num_pig = iu.number_image(pig,1)
    testing_printer.info("pig should be labeled '1'")
    num_linear = iu.number_image(linear,2)
    testing_printer.info("linear should be labeled '2'")

    iu.quick_image_view(pig,title='pig')
    testing_printer.info('PIG', iu.Summarizer(pig) )
    iu.quick_image_view(linear,title='linear')
    testing_printer.info('Linear', iu.Summarizer(linear) )

@iu.human_test
def test_imageloader_resizer_color2gray_viewer_orb_pipeline():
    import imsciutils as iu
    import numpy as np
    testing_printer = iu.get_printer('imageloader->resizer->color2gray->viewer->orb')
    ORB_KEYPOINTS = 10
    # creating all the blocks for the pipeline
    image_loader = iu.ImageLoader()
    resizer = iu.Resizer(to_height=512,to_width=512)
    color2gray = iu.Color2Gray('rgb')
    viewer = iu.BlockViewer()
    orb = iu.Orb(n_keypoints=ORB_KEYPOINTS)

    # creating pipeline with all blocks
    pipeline = iu.Pipeline(name=None,
                            blocks=[image_loader,resizer,viewer,color2gray,orb])


    # getting sample data for this system
    standard_image_filenames = iu.standard_image_filenames()
    processed = pipeline.process(standard_image_filenames)

@iu.human_test
def test_cameracapture_viewer_pipeline():
    import imsciutils as iu
    testing_printer = iu.get_printer('cameracapture->viewer')

    capture = iu.CameraBlock(mode='time')
    viewer = iu.BlockViewer()

    pipeline = iu.Pipeline()
    pipeline.add(capture)
    pipeline.add(viewer)


    #capture for 30 seconds
    images = pipeline.process( [30] )






def main(verbose=False):
    """
    runs all other function in this file automatically and prints out success
    or failure
    """
    import imsciutils as iu
    import six

    if verbose:
        if six.PY2:
            print('verbose options are not available in python2')
        else:
            global VERBOSE
            VERBOSE = True
    else:
        iu.disable_all_printers()

    import sys
    unit_tests = [var for var in globals().values() if callable(var)]
    success = []
    for test_func in unit_tests:
        if test_func.__name__ == 'main':
            # skipping main function to avoid recursion loop
            continue
        else:
            success.append( test_func() )

    # Exit with a 1 code if more than 1 unit test failed
    sys.exit( not all(success) )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet-mode',
                        help='whether or not to print out the arguments passed into functions that use Tester',
                        action='store_true')
    args = parser.parse_args()
    main(not args.quiet_mode)
