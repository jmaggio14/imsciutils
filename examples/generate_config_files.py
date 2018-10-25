"""
Every come up with a situation where you have have a lot of parameters



This is a script  config files using the imsciutils package
created by Jeff Maggio, Ryan Hartzell, Nathan Dileas
https://github.com/jmaggio14/imsciutils

To use this script, simply modify the 'CONFIGS' variable with all variations
of a parameter you want to test. This
"""

import imsciutils as iu
import yaml

BASENAME = 'config'

# if true, then the config files will be named after the config values
USE_EXPLICIT_NAMING = True

# add the config permutations here
# each variable should be a list with all variations of that parameter that you want to test
# for example:
# CONFIGS = {'learning_rate':[.001, .01, .1],
#             'num_layers': [2,3,4],
#             'momentum':[.8,.9],
#             }
CONFIGS = {'learning_rate':[.001, .01, .1],
            'num_layers': [2,3,4],
            'momentum':[.8,.9],
            }

# build a config permutator using imsciutils
permutation_generator = iu.util.Permuter(**CONFIGS)

# iterating through all config permutations
index = 1
for _,configs in permutation_generator:
    # each loop will give us another permuation of our configs
    # so for example, this will generate a dictionary called 'configs'
    # with the following
    # first iteration: configs --> {'learning_rate':.001, 'num_layers':2, 'momentum':.8 }
    # second iteration: configs --> {'learning_rate':.001, 'num_layers':2, 'momentum':.9 }
    # third iteration: configs --> {'learning_rate':.001, 'num_layers':3, 'momentum':.8 }
    # etc...
    # configs change with each iteration

    # constructing filename
    filename = BASENAME + str(index)
    if USE_EXPLICIT_NAMING:
        filename = filename + '--' + ';'.join( '{}={}'.format(k,v) for k,v in configs.items() )
        # checking the length of the filename and making sure it's not to long for windows
        filename = filename[:max(150,len(filename))]


    # checking if the file already exists
    filename = filename + '.yaml'
    filename = iu.io.prevent_overwrite(filename)

    iu.comment("building '{}' with the following kwargs".format(filename) + ''.join( '\n\t{} : {}'.format(k,v) for k,v in configs.items() ))
    iu.info(permutation_generator)
    # saving the config file
    with open(filename,'w') as f:
        file_str = yaml.dump(configs, default_flow_style=False)
        f.write(file_str)

    index += 1
