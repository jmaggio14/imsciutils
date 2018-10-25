![logo](https://github.com/jmaggio14/imsciutils/blob/develop/docs/images/logo.png "logo")
# imagepypelines

![build](https://www.travis-ci.com/jmaggio14/imsciutils.svg?branch=master "master build success")


The `imagepypelines` package consists of high level tools which simplify the construction of complex image processing, computer vision, and machine learning frameworks. During our time in the undergrad Imaging Science program at the Rochester Institute of Technology, we found ourselves writing and rewriting code for things as simple as data type casting and displaying imagery when debugging, causing more trouble than mathematical or logical bugs themselves! Our hope is that the plug-and-play, easily-customizable nature of `imagepypelines` will allow all data-driven scientists to construct complex frameworks quickly for prototyping applications, and serve as a valuable educational tool for those interested in learning traditionally tough subject matter in a friendly environment!

To achieve this goal, our development team always adheres to the following 5 core principles:

1. Legos are fun 
2. Coding should be fun 
3. Therefore coding should be like playing with Legos
4. Imagery is fun, so that will always be our focus
5. We must suffer, lest our users suffer


## Installation
Python compatibility:  3.5+ (Python 2.7 backwards)
via pip:
```
*placeholder until we get it on pypi*
```
from source:
```console
git clone https://github.com/jmaggio14/imagepypelines.git
cd imagepypelines
python setup.py install
```
### dependencies
for full functionality, imsciutils requires _opencv_ and _tensorflow_ to be installed
on your machine
##### tensorflow
if you have a gpu
```console
pip install tensorflow-gpu --user
```
otherwise
```console
pip install tensorflow --user
```
##### opencv
we strongly recommend that you [build opencv from source](https://docs.opencv.org/3.4/df/d65/tutorial_table_of_content_introduction.html)

**_however_** unofficial bindings for opencv can be installed with
```console
pip install opencv-python --user
```
(while we haven't encountered many problems with these unofficial bindings,
we do not guarantee support)


## Documentation
Full documentation for `imagepypelines` can be found on our website: www.imagepypelines.org


## Licensing / Credit
`imagepypelines` is licensed under the [MIT](https://choosealicense.com/licenses/mit/) permissive software license. You may use this code for commercial or research use so long as it conforms to the terms of the license included in this repo as well as the licenses of `imagepypelines` dependencies.

Please credit us if you use `imagepypelines` in your research
```
*placeholder for latex*
```


# What Makes Us Unique?

## The Pipeline
`imagepypelines`'s most powerful feature is a high level interface to create data processing pipelines which apply a sequence of algorithms to input data automatically.

In our experience as imaging scientists, processing pipelines in both corporate or academic settings are not always easy to adapt for new purposes and are therefore too often relegated to _proof-of-concept_ applications only. Many custom pipelines may also not provide step-by-step error checking, which can make debugging a challenge.


![xkcd](https://imgs.xkcd.com/comics/data_pipeline.png "cracked pipelines")

(source: [XKCD](https://www.xkcd.com/2054/))


The `Pipeline` object of `imagepypelines` allows for quick construction and prototyping, ensures end-to-end compatibility through each layer of a workflow, and leverages helpful in-house debugging utilities for use in image-centric or high-dimensional data routines.


## The Block
write some more stuff here about purpose of blocks as related to pipelines and then go into how to actually build a pipeline


## Building a pipeline
Pipelines in `imsciutils` are constructed of processing `blocks` which apply an algorithm to a sequence of data passed into it.

![pipeline](https://github.com/jmaggio14/imsciutils/blob/develop/docs/images/pipeline-example.png "pipeline example")

Each `block` _takes in_ a list of data and _returns_ a list of data, passing it onto the next block or out of the pipeline. This system ensures that blocks are compatible with algorithms that process data in batches or individually. Blocks also support label handling, and thus are **compatible with supervised machine learning systems or other algorithms that require training**

##### let's create an example pipeline
let's say we want a system that reads in images, resizes them, and then displays them for us
```python
import imsciutils as iu

# first let's create our blocks
block1 = iu.ImageLoader()
block2 = iu.Resizer(512,512)
block3 = iu.BlockViewer(pause_time=1)

# then build a pipeline
pipeline = iu.Pipeline(blocks=[block1,block2,block3])

# we'll use imsciutils example data
image_filenames = iu.standard_image_filenames()

# now we process it!
pipeline.process(image_filenames)
```
Now we have a system the reads in and displays imagery!

But what if we want to do something more complicated? Let's say we want to apply a lowpass filter to all of these images before we display them?
```python
import imsciutils as iu

# first let's create our blocks
load = iu.ImageLoader()
resize = iu.Resizer(512,512)
fft = iu.FFT()
lowpass = iu.Lowpass(cut_off=32)
ifft = iu.IFFT()
display = iu.BlockViewer(pause_time=1)

# then build a pipeline
pipeline = iu.Pipeline(blocks=[load,resize,fft,lowpass,ifft,display])

# we'll use imsciutils example data
image_filenames = iu.standard_image_filenames()

# now we process it!
pipeline.process(image_filenames)
```

### Machine Learning Applications
One of the more powerful applications of `imsciutils` is it's ease of use in
_machine learning_ and _feature engineering_ applications. We can easily build
a simple image classifier that is tailored to your purposes

```python
import imsciutils as iu

features = iu.PretrainedNetwork() # generate features
neural_network = iu.MultilayerPerceptron(neurons=512) # NN classifier
# there are a lot more parameters you can tweak!

classifier = iu.Pipeline([features,neural_network])

# for this example, we'll need to load the standard Mnist handwriting dataset
# built into `imsciutils`
mnist = iu.Mnist()
train_data, train_labels = mnist.get_train()
test_data, ground_truth = mnist.get_test()

# train the classifier
classifier.train(train_data,train_labels)

# test the classifier
predictions = classifier.process(test_data)

# print the accuracy
accuracy = iu.accuracy(predictions,ground_truth)
print(accuracy)
```



#### builtin classifiers
- Multilayer Perceptron
- Linear Support Vector Machine



#### builtin I/O blocks
- Webcam Capturing
- Image Loading
- Image Display

#### builtin utility blocks
- Image Resizing
- Grayscale Conversion

#### builtin feature engineering blocks include:
- keypoint detection and description
- Fourier Transform
- Frequency Filtering
- Pretrained Neural Networks for image feature generations

### Designing your own processing blocks
Designing your blocks is a fairly straightforward process
#### Quick Block Creations
if you already have a function that does all the processing you need to a single input,
then you can  

## chaining multiple pipelines together


# Imaging Science Convenience Functions
## Getting Standard Test Imagery
`imsciutils` contains helper functions to quickly retrieve imagery that
are frequently used as benchmarks in the Imaging Science community
```python
import imsciutils as iu
lenna = iu.lenna()
linear_gradient = iu.linear()
```
A full list of standard images can be retrieved with `iu.list_standard_images()`

for those of you in the Imaging Science program at RIT, there are a
couple easter eggs for ya ;)
```python
import imsciutils as iu
iu.quick_image_view( iu.carlenna() )
iu.quick_image_view( iu.roger() )
iu.quick_image_view( iu.pig() )
```


## Viewing Imagery
Viewing imagery can be an surprisingly finicky process that differs machine
to machine or operating over X11. `imsciutils` contains helper functions and objects for this purpose

### quick image viewer:

when you want to quickly display an image without any bells and whistles,
you can use the `quick_image_view` function

```python
import imsciutils as iu
lenna = iu.lenna()

# Now lets display Lenna
iu.quick_image_view( lenna )

# display lenna normalized to 255
iu.quick_image_view(lenna, normalize_and_bin=True)
```

### Robust Image Viewer:

When you want a tool that can display multiple images at once, resize
images when desired and an optional frame_counter, you can use the `Viewer` object
```python
import imsciutils as iu
import time

# lets build our Viewer and have it auto-resize images to 512x512
viewer = iu.Viewer('Window Title Here', size=(512,512))
# let's enable the frame counter, so we know what image we are on
viewer.enable_frame_counter()

# get all standard images
standard_images = iu.standard_image_gen()

# now let's display all images sequentially!
for img in standard_images:
	viewer.view( img )
	time.sleep(.1)
```

### Normalizing and binning an image
forgetting to do this gets ya more often than you might think when displaying
an image

```python
import imsciutils as iu
import numpy as np
random_pattern = np.random.rand(512,512).astype(np.float32)

display_safe = iu.normalize_and_bin(random_pattern)
iu.Viewer().view(display_safe)
```
### Array Summarization
when debugging an image pipeline, printing out an image
can be counter productive. Imaging scientists frequently default
to printing out the shape or size of the data. `imsciutils` contains
a helper class to quickly summarize an image in a formatted string
```python
import imsciutils as iu
lenna = iu.lenna()

summary = iu.Summarizer(lenna)
print(summary)
```
produces the following
```
'[ARRAY SUMMARY | shape: (512, 512, 3) | size: 786432 | max: 255 | min: 3 | mean: 128.228 | dtype: uint8]'
```

### Image Coordinates
helper functions to get image coordinates quickly, useful if your
applications involve a mix of color and grayscale images.
Mostly useful to clean up code and avoid silly mistakes
```python
import imsciutils as iu
lenna = iu.lenna()

# center pixel in the image
center_row, center_col = iu.centroid(lenna)

# number of rows and columns
rows, cols = iu.frame_size(lenna)

# shape and dtype
rows, cols, bands, dtype = iu.dimensions(lenna)
```

### Timing
Many imaging tasks are time sensitive or computationally
intensive. `imsciutils` includes simple tools to time your process or function

#### Timer Objects
`imsciutils` also includes a separate timer for timing things inside a function
or code block

##### absolute timing:
```python
from imsciutils.util import Timer
import time

t = Timer()
time.sleep(5)
print( t.time(),"seconds" ) # or t.time_ms() for milliseconds
```

##### lap timing:
```python
from imsciutils.util import Timer
import time

t = Timer()
for i in range(10):
	time.sleep(1)
	print( t.lap(),"seconds" ) # or t.lap_ms() for milliseconds
```

##### perform operation for N seconds:
```python
from imsciutils.util import Timer
import time

def do_something():
	pass

# set the countdown
N = 10 #seconds
t = Timer()
t.countdown = N
while t.countdown:
	do_something()
```


#### timing Decorator
let's say we have a function that we think may be slowing down our pipeline.
We can add `@function_timer` on the line above the function
and see it automatically print how long the function took to run
```python
from imsciutils.util import function_timer
from imsciutils.util import function_timer_ms
import time

# add the decorator here
@function_timer
def we_can_time_in_seconds():
	time.sleep(1)

# we can also time the function in milliseconds using '@function_timer_ms'
@function_timer_ms
def or_in_milliseconds():
	time.sleep(1)

we_can_time_in_seconds()
or_in_milliseconds()
```
prints the following when the above code is run
```
(  function_timer  )[    INFO    ] ran function 'we_can_time_in_seconds' in 1.001sec
(  function_timer  )[    INFO    ] ran function 'or_in_milliseconds' in 1000.118ms
```

# Development Tools in `imsciutils`
**_This section is for developers of `imsciutils` or people who want `imsciutils` closely integrated with their projects_**

## Printers
Are you a scientist???
If so, then you probably use millions of print statements to debug your code. (don't worry, we are all guilty of it)

`imsciutils` encourages code traceability through the use of an object known as a **`Printer`**. Printers are objects that simply print out what's happening in a manner that's easy to read, color coded, and traceable to the object that is performing the current action. **Printers are extremely low overhead and will not affect the speed of your code more than a print statement.**

The functionality is similar to python's [`logging`](https://docs.python.org/3.7/library/logging.html) module

### making printers
printers can be created or retrieved using the `get_printer` function
```python
import imsciutils as iu
printer = iu.get_printer('name your printer here')
```

### printer levels
printer messages can be filtered be priority so that only desired messages can be seen. In `imsciutils`, printer levels are also color coded so they can be read easily in a console
```python
import imsciutils as iu

example_printer = iu.get_printer('example!')
example_printer.debug('message') # prints 'message' at level 10 - blue text
example_printer.info('message') # prints 'message' at level 20 - white text
example_printer.warning('message') # prints 'message' at level 30 - yellow text
example_printer.comment('message') # prints 'message' at level 30 - green text
example_printer.error('message') # prints 'message' at level 40 - red text
example_printer.critical('message') # prints 'message' at level 50 - bold red text
```
Any level that is less than the current `GLOBAL_LOG_LEVEL` will **NOT** be printed. This makes it easy to filter out statements which may be erroneous or too numerous to make sense of.

this value can be set with the `set_global_printout_level` function
```python
import imsciutils as iu
iu.set_global_printout_level('warning') # debug and info statements will not print now
```
local printer levels can be set with `Printer.set_log_level`
```python
import imsciutils as iu
printer = iu.get_printer('Example Printer')
printer.set_log_level('error') # only error and critical functions will print
```

(this system is exactly the same as log_levels in python's [`logging`](https://docs.python.org/3.7/library/logging.html) module )

### disable or enabling certain printers
Sometimes you may only want to see printouts from a specific class or function. you can do this
with the `whitelist_printer`, `blacklist_printer`, or `disable_all_printers` functions

### default printer
there's a default printer in `imsciutils` which is accessible through functions in the main module
```python
iu.debug('debug message') # level=10 --> (    imsciutils    )[    DEBUG    ] debug message
iu.info('info message') # level=20 --> (    imsciutils    )[    INFO    ] debug message
iu.warning('warning message') # level=30 --> (    imsciutils    )[    WARNING    ] warning message
iu.error('error message') # level=40 --> (    imsciutils    )[    ERROR    ] error message
iu.critical('critical message') # level=50 --> (    imsciutils    )[    CRITICAL    ] critical message
iu.comment('comment message') # level=30 --> (    imsciutils    )[    COMMENT    ] comment message
```

### class printers
a good strategy to encourage traceability is to create a printer object as a class instance attribute
```python
import imsciutils as iu

class ExampleClass(object):
	def __init__(self,*args,**kwargs):
		name_of_class = self.__class__.__name__
		self.printer = iu.get_printer(name_of_class)
		self.printer.info("object instantiated!")

		self.do_something()

	def do_something(self):
		self.printer.warning("did something!")

ExampleClass()
```
produces the following
```
(   ExampleClass   )[    INFO    ] object instantiated!
(   ExampleClass   )[   WARNING  ] did something!
```
This way it's easy track what stage of the pipeline your code is in, because each object will have it's own printer and be distinguishable in the terminal!

## development decorators
`imsciutils` contains four decorators that are made for use by developers in the backend

### @deprecated
made to decorate functions or classes that are deprecated
```python
import imsciutils as iu

@iu.deprecated("'old_function' has been renamed to 'new_function'. references will be removed in a future version!")
def old_function():
	pass # real code will do something

old_function()
```

produces the following
```
(    imsciutils    )[   WARNING  ] DEPRECIATION WARNING: 'old_function' has been renamed to 'new_function'. references will be removed in a future version!
```
### @experimental
made to decorate functions or classes that are experimental and may not be fully tested yet
```python
import imsciutils as iu

@iu.experimental() # you can include a custom message here if you want
def new_feature():
	pass

new_feature()
```
produces the following
```
(    imsciutils    )[   WARNING  ] EXPERIMENTAL WARNING: 'new_feature' is an experimental feature
```

### @human_test
This is a decorator made for unit tests which require a human to verify functionality. (for example: functions that display images)

**WARNING: unlike most decorators, this will not return the output of the wrapped function, but instead True or False.
This is because it is meant for Unit Tests, NOT actual use in a pipeline**
```python
import imsciutils as iu
@iu.human_test
def unit_test_for_quick_image_view():
	iu.quick_image_view( iu.lenna() )

print('test succeeded ': unit_test_for_quick_image_view())
```
_lenna will display_

produces the following the terminal for the user to answer
```
did the test for 'unit_test_for_quick_image_view' succeed? Yes? No?
```


### @print_args
Decorator to print out the arguments a function is running with. Unlike other decorators described here, we encourage you to use this decorator frequently in your code during development to avoid silly mistakes
```python
import imsciutils as iu
@iu.print_args
def func_with_lots_of_args(a, b, c=3, d=4):
			pass
func_with_lots_of_args(1, b=2, c='not 3')
```
produces the following in the terminal
```
(dimensions Tester)[    INFO    ] running 'dimensions' with the following args:
	positional    | img            : [ARRAY SUMMARY | shape: (512, 512, 3) | size: 786432 | max: 255 | min: 3 | mean: 128.228 | dtype: uint8]
	default       | return_as_dict : False
```
