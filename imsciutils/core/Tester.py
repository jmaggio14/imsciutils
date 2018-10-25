# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
import numpy as np
import inspect
from collections import Iterable, OrderedDict
import collections
from .Printer import get_printer
from .. import util
from .debug import debug
from .Summarizer import Summarizer
from .error_checking import is_numpy_array
from .printout import warning as iuwarning
from .printout import info as iuinfo
import six


class Tester(object):
    """
    Testing object used to simply unit testing for a target function. This object
    can perform exact and type check tests on a target function and interally

    Args:
        target (callable): function target to test
        verbose (bool): whether or not to be verbose
    """
    def __init__(self, target, verbose=True):
        if not callable(target):
            error_msg = "'target' must be a callable class or function!"
            raise TypeError(error_msg)

        self.target = target
        self.verbose = verbose
        self.printer = get_printer(target.__name__ + " Tester")


    def __str__(self):
        return "[imsciutils Tester for '{}' function]".format(self.target.__name__)

    def __repr__(self):
        return self.__str__()

    def exact_test(self, desired_output, *args, **kwargs):
        """
        Checks whether the target function outputs the exact output given
        by 'desired_output'

        Args:
            desired_output (variable): the exact output the target
                function should return
            *args: positional arguments for the target
            **kwargs: keyword arguments for the target

        Returns:
            bool: whether or not the output is the same as the
                desired_output
        """
        # printing out args and kwargs if desired by user
        if self.verbose:
            self.__print_args(*args,**kwargs)


        # running the target with a debug wrapper
        out = self.__run_target(*args,**kwargs)

        # testing the output
        if is_numpy_array(out) and is_numpy_array(desired_output):
            has_failed = not np.all(out == desired_output)
        else:
            has_failed = not np.all( np.array(out == desired_output) )


        if has_failed:
            # TEST HAS FAILED
            # converting to any numpy arrays to summaries for printout
            if is_numpy_array(desired_output):
                desired_output = Summarizer(desired_output)
            if is_numpy_array(out):
                out = Summarizer(out)

            self.printer.error("{} test failure expected output {}, but got {}"\
                            .format(self.target.__name__,desired_output, out))
            return False
        else:
            # TEST HAS SUCEEDED
            self.printer.comment("{} exact test success!".format(self.target.__name__))
            return True


    def type_test(self,desired_type,*args,**kwargs):
        """
        Checks whether the target function outputs the correct type or types
        given a set of args and kwargs

        Args:
            desired_type (type,Iterable of types): the type or types
                the output function should return
            *args: positional arguments for the target
            **kwargs: keyword arguments for the target

        Returns:
            passed (boolean): whether or not the output was one of the desired_type

        Example:
            >>> def output_str_or_int(a):
            ...    if a:
            ...        return "1"
            ...    else:
            ...        return 1

            >>> tester = Tester(output_str_or_int)
            >>> is_test_successful = tester.type_test(str, a=True)
        """
        # Making desired_type a list if it isn't already
        if not isinstance(desired_type,Iterable):
            desired_type = [desired_type]

        # print out args and kwargs if desired
        if self.verbose:
            self.__print_args(*args,**kwargs)

        # run the target function
        out = self.__run_target(*args,**kwargs)

        # testing the output
        if not isinstance(out,tuple(desired_type)):
            # TEST HAS FAILED
            self.printer.error("{} test failure expected output {}, but got {}"\
                        .format(self.target.__name__,desired_type, type(out)))

            return False
        else:
            # TEST HAS SUCEEDED
            self.printer.comment("{} type test successful! with type {}!"\
                            .format(self.target.__name__,type(out)))
            return True


    def custom_test(self,test_func,*args,**kwargs):
        """
        function to build custom tests using an input function
        that will evaluate the output of the target

        Args:
            test_func (callable): the function to evaluate
            *args: positional arguments for the target
            **kwargs: keyword arguments for the target

        """
        if self.verbose:
            self.__print_args(*args,**kwargs)

        if not callable(test_func):
            error_msg = "'test_func' must be a function or callable object"
            self.printer.error(error_msg)
            raise TypeError(error_msg)

        passed = test_func( self.target(*args,**kwargs) )
        return passed





    def __print_args(self,*args,**kwargs):
        """
        prints the arguments passed into the target
        """
        POSITIONAL    = 'positional    |'
        KEYWORD       = 'keyword       |'
        VARPOSITIONAL = 'var-positional|'
        VARKEYWORD    = 'var-keyword   |'
        DEFAULT       = 'default       |'

        arg_dict = collections.OrderedDict()
        vtypes = {}
        def __add_to_arg_dict(key,val,vtype):
            if is_numpy_array(val):
                val = str( Summarizer(val) )
            arg_dict[key] = val
            vtypes[key] = vtype

        if six.PY2:
            # JM:
            # skipping this function is using python2 because
            # inspect.getfullargspec is python3+ only
            self.printer.warning('argument printout is not available in python2')
            return

        spec = inspect.getfullargspec(self.target)
        specdefaults = [] if spec.defaults is None else spec.defaults
        specargs = [] if spec.args is None else spec.args
        speckwonlyargs = [] if spec.kwonlyargs is None else spec.kwonlyargs
        speckwonlydefaults = {} if spec.kwonlydefaults is None else spec.kwonlydefaults

        num_positional_passed_in = len(args)
        num_required = len(specargs) - len(specdefaults)

        # adding default positional args values to the dictionary
        for i,var_name in enumerate(specargs):
            if i < num_required:
                var = util.red("No argument was passed in!",bold=True)
            else:
                var = specdefaults[i - num_required]

            vtype = DEFAULT
            __add_to_arg_dict(var_name,var,vtype)

        # positional arguments passed in and varargs passed in
        for i in range(num_positional_passed_in):
            if i < num_required:
                var_name = specargs[i]
                vtype = POSITIONAL
            else:
                var_name = 'arg{}'.format(i)
                vtype = VARPOSITIONAL
            var = args[i]
            __add_to_arg_dict(var_name,var,vtype)

        # adding keyword only args to the dict
        for var_name in speckwonlyargs:
            var = util.red("No argument was passed in!",bold=True)
            vtype = KEYWORD
            __add_to_arg_dict(var_name,var,vtype)
        for var_name,var in speckwonlydefaults.items():
            vtype = DEFAULT
            __add_to_arg_dict(var_name,var,vtype)

        # keyword arguments passed in
        for var_name in kwargs:
            if var_name in specargs:
                vtype = KEYWORD
            else:
                vtype = VARKEYWORD
            var = kwargs[var_name]
            __add_to_arg_dict(var_name,var,vtype)

        # formatting the actual string to be printed out
        self.printer.info("running '{}' with the following args:".format(self.target.__name__))
        if len(arg_dict) == 0:
            __add_to_arg_dict('None','','')
        longest_arg_name = max(len(k) for k in arg_dict)
        arg_string = ''.join(["\t{} {} : {}\n".format(vtypes[k], k+(' ' * (longest_arg_name-len(k))), v) for k,v in arg_dict.items()])
        print( arg_string )

    def __run_target(self,*args,**kwargs):
        # testing to make sure the function will run
        try:
            out = self.target(*args, **kwargs)
            return out
        except Exception as e:
            self.printer.error("{} test failed to run!".format(self.target.__name__))
            debug(e)



def main():
    def test_target(a, b, c, ryanisdumb='yes'):
        return a

    tester = Tester(test_target)
    desired_output = False
    tester.exact_test(desired_output, True, iu.lenna(), None)
    tester.type_test(bool, True, iu.lenna(), None, ryanisdumb="yes")


if __name__ == "__main__":
    main()
