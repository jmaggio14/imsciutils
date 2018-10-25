# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .printout import warning as iuwarning
from .printout import info as iuinfo
from .. import util
from .error_checking import is_numpy_array
from .Printer import get_printer


import six
from termcolor import colored
import inspect
import collections

def deprecated(depreciation_msg):
    """Decorator for functions that are deprecated but still extant,

    Args:
        depreciation_msg (str): msg to print to the terminal when this
            function runs.

    Example:
        >>> import imsciutils as iu
        >>> @iu.deprecated("custom depreciation message here")
        >>> def new_feature():
        ...    pass

    """
    def _deprecated(func):
        def __deprecated(*args,**kwargs):
            iuwarning("DEPRECIATION WARNING:", depreciation_msg)
            return func(*args,**kwargs)

        return __deprecated
    return _deprecated



def experimental(experimental_msg=None):
    """
    Decorator for functions that are considered experimental,

    Args:
        experimental_msg (str, None): msg to print to the terminal when this
            function runs

    Example:
        >>> import imsciutils as iu
        >>> @iu.experimental("optional message - you can leave blank")
        >>> def new_feature():
        ...    do_something()

    """
    create_message = False
    if experimental_msg is None:
        create_message = True

    def _experimental(func):
        def __experimental(*args,**kwargs):
            if create_message:
                experimental_msg = "'{}' is an experimental feature".format(func.__name__)
            iuwarning("EXPERIMENTAL WARNING:", experimental_msg)
            return func(*args,**kwargs)

        return __experimental
    return _experimental



def human_test(func):
    """
    Decorator for unit tests which require human interaction to verify their success,
    such as tests that perform complicated segmentation, classification, graphing
    or interaction with displays

    Args:
        func (callable): function or other callable to wrap in a unit test

    Example:
        >>> import imsciutils as iu
        >>> @iu.human_test # no parantheses are needed!
        >>> def function_that_displays_something():
        ...    do_something()

    """
    query_string = "did the test for '{}' succeed? {Y}es? {N}o?\n".format(func.__name__,
                                                                    Y=colored('Y',attrs=['bold']),
                                                                    N=colored('N',attrs=['bold']))
    def _ask_input():
        if six.PY2:
            out = raw_input(query_string)
        else:
            out = input(query_string)

        if out.lower() in ['yes','y']:
            return True
        elif out.lower() in ['no','n']:
            return False
        else:
            return _ask_input()

    def _human_test(*args,**kwargs):
        ret = func(*args,**kwargs)
        if _ask_input():
            return True
        else:
            return False

    return _human_test


def print_args(func):
    """decorator to print out the arguments that a function is running with,
    this includes: arguments passed in, default values that are unspecified,
    varargs (*args), and varkwargs (**kwargs)

    Args:
        func (callable): function or callable to print input arguments of

    Example:
        >>> import imsciutils as iu
        >>> @iu.print_args
        >>> def func_with_lots_of_args(a, b, c=3, d=4):
        ...    pass
        >>> func_with_lots_of_args(1, b=2, c='not 3')
        >>> # produces the following in the terminal
        (  positional  ) a : 1
        (   keyword    ) b : 2
        (   keyword    ) c : not 3
        (  positional  ) d : 4
    """

    def _print_args(*args,**kwargs):
        """
        prints the arguments passed into the target
        """
        POSITIONAL    = 'positional    |'
        KEYWORD       = 'keyword       |'
        VARPOSITIONAL = 'var-positional|'
        VARKEYWORD    = 'var-keyword   |'
        DEFAULT       = 'default       |'
        PRINTER = get_printer(func.__name__)

        arg_dict = collections.OrderedDict()
        vtypes = {}
        def __add_to_arg_dict(key,val,vtype):
            if is_numpy_array(val):
                val = str( Summarizer(val) )
            arg_dict[key] = val
            vtypes[key] = vtype


        spec = inspect.getfullargspec(func)
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
        PRINTER.info("running '{}' with the following args:".format(func.__name__))
        if len(arg_dict) == 0:
            __add_to_arg_dict('None','','')
        longest_arg_name = max(len(k) for k in arg_dict)
        arg_string = ''.join(["\t{} {} : {}\n".format(vtypes[k], k+(' ' * (longest_arg_name-len(k))), v) for k,v in arg_dict.items()])
        print( arg_string )

        ret = func(*args,**kwargs)
        return ret
    return _print_args





def unit_test(func):
    """
    Decorator which prints a colored message
    """
    def _unit_test(*args,**kwargs):
        passed = func(*args,**kwargs)

        if passed:
            msg = util.green("{} test passed!".format(func.__name__))
        else:
            msg = util.red("{} test failed!".format(func.__name__))

        print(msg)
        return passed

    return _unit_test
