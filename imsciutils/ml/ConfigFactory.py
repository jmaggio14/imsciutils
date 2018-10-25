# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from collections import Iterable
import itertools
import numpy as np

class ConfigFactory(object):
    """
    argument Permuter object for generating permutations of function arguments
    or configurations for config files

    For example, in many machine learning applications, parameters have to
    be tweaked frequently to optimize a model. This can be a tedious task
    and frequently involves a human tweaking configurations files. This
    object is meant to simplify that process by generating permutations
    from a sample of arguments and keyword arguments

    Example:
        >>> def run_important_test(arg1,arg2,arg3,first,second,third):
        ...    do_something_important()


        >>> arg_trials = [
        ...        [1,2,3], # trials for first positional argument
        ...        ['a','b','c'], # trials for second positional arguments
        ...        ['y','z'], # trials for third positional argument
        ...        ]

        >>> kwarg_trials = {
        ...            'first':None, # trials for 'first' keyword argument
        ...            'second':['I','J','K'], # trials for 'second' keyword argument
        ...            'third':['i','j','k'], # trials for 'third' keyword argument
        ...            }

        >>> permuter = ConfigFactory(*arg_trials,**kwarg_trials)
        >>> for args,kwargs in permuter:
        ...    run_important_test(*args,**kwargs)

    """
    def __init__(self,*arg_trials,**kwarg_trials):
        arg_list = []
        self.num_positional = len(arg_trials)

        # making positional arguments iterable if they aren't already
        for arg in arg_trials:
            if not isinstance(arg,Iterable):
                arg = [arg]
            arg_list.append(arg)

        # making keyword arguments iterable if they aren't already
        for key,val in kwarg_trials.items():
            if not isinstance(val,Iterable):
                kwarg_trials[key] = [val]

        self.kwarg_keys = sorted( kwarg_trials.keys() )
        kwarg_vals = [kwarg_trials[k] for k in self.kwarg_keys]

        # storing all arguments in a list
        all_args = arg_list + kwarg_vals
        self.num_permutations = int( np.prod([ len(args) for args in all_args ] ) )
        self._remaining = self.num_permutations

        # generation of permutations using a cartesian product
        self.permutations = itertools.product(*all_args)

    def __iter__(self):
        """
        iterates through the permutations
        """
        self._remaining = self.num_permutations
        #retrieving all permutations as a generator
        for perm in self.permutations:
            args = perm[:self.num_positional]
            kwargs = dict( zip(self.kwarg_keys,perm[self.num_positional:]) )
            yield args,kwargs
            self._remaining -= 1

    def __len__(self):
        """returns the number of total permutations"""
        return self.num_permutations

    def remaining(self):
        """returns the number of remaining permutations"""
        return self._remaining

    def __str__(self):
        num_remaining = "unknown"
        if hasattr(self,'_remaining'):
            num_remaining = self.remaining()
        out = "ConfigFactory ({} permutations remaining)".format(num_remaining)
        return out

    def __repr__(self):
        return str(self) + ' @{}'.format( hex(id(self)) )


def main():
    arg_trials = [
            [1,2,3],
            ['a','b','c'],
            ['y','z'],
            ]

    kwarg_trials = {
                'first':None,
                'second':['I','J','K'],
                'third':['i','j','k'],
                }

    perm_gen = ConfigFactory(*arg_trials,**kwarg_trials)

    perm_idx = 0
    for args,kwargs in perm_gen:
        print( args,kwargs )
        print( perm_gen.remaining(),'permutation remaining' )
        perm_idx += 1

    print( perm_idx )
    print( len(perm_gen) )

if __name__ == "__main__":
    main()
