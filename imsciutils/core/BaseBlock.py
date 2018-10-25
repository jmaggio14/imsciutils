# @Email: jmaggio14@gmail.com
# @Website: https://www.imagepypelines.org/
# @License: https://github.com/jmaggio14/imsciutils/blob/master/LICENSE
# @github: https://github.com/jmaggio14/imsciutils
#
# Copyright (c) 2018 Jeff Maggio, Nathan Dileas, Ryan Hartzell
from .Printer import get_printer
from .Exceptions import InvalidBlockInputData
from .Exceptions import InvalidProcessStrategy
from .Exceptions import InvalidLabelStrategy
from .Exceptions import DataLabelMismatch
from .Exceptions import BlockRequiresLabels
from .Exceptions import IncompatibleTypes
from .constants import NUMPY_TYPES
import copy


def quick_block(process_fn,
                io_map,
                name=None):
    """convienence function to make simple blocks

    Args:
        process_fn(func): function that takes in and processes
            exactly one datum

        io_map(IoMap,dict): dictionary of input-output mappings for this
            Block
        name(str): name for this block, it will be automatically created/modified
            to make sure it is unique

    Returns:
        block(iu.SimpleBlock): simple block that applies the given function

    Example:
        >>> import imsciutils as iu
        >>> def calculate_orb_features(datum):
        ...     _,des = cv2.ORB_create().detectAndCompute(datum,None)
        ...     return des
        >>>
        >>> block = iu.simple_block(calculate_orb_features,
        ...                         input_shape=[None,None],
        ...                         output_shape=[None,32])
        >>>
    """
    if name is None:
        name = process_fn.__name__

    process_fn = staticmethod(process_fn)
    block_cls = type(name, (SimpleBlock,), {'process': process_fn})
    block = block_cls(io_map=io_map,
                      name=name)
    return block


class ArrayType(object):
    """Object to describe the shapes of Arrays for Block inputs or outputs

    Object that contains the shapes and datatypes of an input or output
    for a Block

    Args:
        *array_shapes(vargs of array shapes): acceptable shapes. Arbitrary
            length axes can be represented by None.
            example: [None,None,3] (for rgb image)
        dtypes(np.dtype,tuple): keyword only argument.
            numpy dtype or dtypes for this input/output. default is NUMPY_TYPES
    """

    def __init__(self, *array_shapes, **dtypes_kwarg):
        # -------------------- error-checking ---------------------
        if not all(isinstance(shape, (tuple, list)) for shape in array_shapes):
            raise TypeError("all array shapes must be tuples or lists")

        # JM: this is stupid hack to get keyword only arguments in python2
        if len(dtypes_kwarg) > 1:
            raise TypeError("only one keyword argument 'dtypes' can be specified")
        if len(dtypes_kwarg) >= 1 and ('dtypes' not in dtypes_kwarg):
            raise TypeError("only one keyword argument 'dtypes' can be specified")


        if 'dtypes' in dtypes_kwarg:
            dtypes = dtypes_kwarg['dtypes']
        else:
            dtypes = NUMPY_TYPES

        # check to see if dtypes is a single valid numpy dtype
        if dtypes in NUMPY_TYPES:
            dtypes = (dtypes,)
        # otherwise it must be a tuple or list of values in NUMPY_TYPES
        elif isinstance(dtypes, (list, tuple)):
            if not all((dt in NUMPY_TYPES) for dt in dtypes):
                raise ValueError("dtypes must be None or a tuple/list of valid numpy dtypes")
        # otherwise we've recieved an input that doesn't make sense
        else:
            raise TypeError("dtypes must be None or a tuple/list of valid numpy dtypes")

        # -------------------- real code begins ---------------------
        array_shapes = tuple(tuple(shp) for shp in array_shapes)
        self.shapes = array_shapes
        self.dtypes = dtypes

    def __str__(self):
        if self.dtypes == NUMPY_TYPES:
            return "ArrayType({}, dtypes=any)"\
                .format(', '.join(str(s) for s in self.shapes))
        else:
            return "ArrayType({}, dtypes='{}')"\
                .format(', '.join(str(s) for s in self.shapes), self.dtypes)

    def __repr__(self):
        return str(self)

    def __eq__(self,other):
        if isinstance(other,ArrayType):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash(self.shapes + self.dtypes)

# acceptable types for datums passed between blocks
BLOCK_VALID_TYPES = [str, int, float, None, ArrayType]
BLOCK_NON_ARRAY_TYPES = [str, int, float, None, ]


class IoMap(tuple):
    """mapping object to determine the output of block

    IoMaps are used to predict the output of a block given a certain type of
    input. Every block contains an IoMap located under block.io_map

    Args:
        io_map(dict,IoMap): dictionary that describes the input outputs of the
            block. Of the form: io_map[input_type] = output_type

    Attributes:
        non_arrays(tuple): mapping of non-array input-outputs for this io_map
        arrays(tuple): mapping of array input-outputs for this io_map
        inputs(tuple): tuple of inputs for this io map
        outputs(tuple): tuple of outputs for this io map

    """
    def __new__(cls, io_map):
        # -------------- ERROR CHECKING -----------------------
        if isinstance(io_map, IoMap):
            return copy.copy(io_map)
        elif not isinstance(io_map, dict):
            raise TypeError("IoMap must be instantiated with a dictionary")

        for i, o in io_map.items():
            if not ((i in BLOCK_VALID_TYPES) or isinstance(i, ArrayType)):
                raise TypeError("unacceptable io_map key, must be {}".format(BLOCK_VALID_TYPES))
            if not ((o in BLOCK_VALID_TYPES) or isinstance(o, ArrayType)):
                raise TypeError("unacceptable io_map value, must be {}".format(BLOCK_VALID_TYPES))

        # ---------------- Breaking dictionary up into a mapping --------------
        # {key1:val1,key2:val2} --> ( (key1,val1),(key2,val2) )
        io_map = io_map.items()
        # splitting apart all Array Types with multiple shapes
        # ArrayType(shape1,shape2) --> ArrayType(shape1), ArrayType(shape2)
        # going through inputs first
        reduced_io_map = []
        for i, o in io_map:
            reduced_io_map.extend(cls.reduce(i, o))

        # return the new reduced mapping
        return super(IoMap, cls).__new__(cls, tuple(set(reduced_io_map)))

    def __init__(self, io_map):
        self.non_arrays = tuple((i, o) for i, o in self if (i in BLOCK_NON_ARRAY_TYPES))
        self.arrays = tuple((i, o) for i, o in self if isinstance(i, ArrayType))

        self.inputs = tuple(i for i, o in self)
        self.outputs = tuple(o for i, o in self)

    @staticmethod
    def reduce(i, o):
        """reduces ArrayTypes with multiple shapes to multiple single
        shape ArrayTypes

        ArrayType(shape1,shape2) --> ArrayType(shape1), ArrayType(shape2)

        Args:
            i (ArrayType): input ArrayType
            o (ArrayType): output ArrayType

        Returns:
            reduced(tuple): tuple mapping of reduced types ((i1,o1),(i2,o2)...)
        """
        if i in BLOCK_NON_ARRAY_TYPES:
            reduced_i = ((i, o), )
        else:
            split = tuple(ArrayType(shp,dtypes=i.dtypes) for shp in i.shapes)
            reduced_i = zip(split, (o,)*len(split))

        reduced = []
        for i, o in reduced_i:
            if o in BLOCK_NON_ARRAY_TYPES:
                reduced.append((i, o))
            else:
                split = tuple(ArrayType(shp,dtypes=o.dtypes) for shp in o.shapes)
                reduced.extend(zip((i,)*len(split), split))

        return tuple( reduced )

    @staticmethod
    def shape_comparison(input_shape, acceptable_shape):
        """compares ArrayType shapes and returns a boolean to indicate
        compatability

        Args:
            input_shape(tuple): ArrayType shape of the input data
            acceptable_shape(tuple): one of the acceptable input type array
                shapes

        Returns:
            compatible(bool): whether or the input shape is compatible with
                this block
        """
        # if they have a different number of axis, they aren't compatible
        if len(input_shape) != len(acceptable_shape):
            return False

        # compare every element
        compatible_by_axis = []
        for input_i, acceptable_i in zip(input_shape, acceptable_shape):
            # if block element is None, then arbitrary length for this axis is accepted
            # so no more comparisons are needed for this element
            if (acceptable_i == None) or (input_i == acceptable_i):
                compatible_by_axis.append(True)
            else:
                compatible_by_axis.append(False)

        return all(compatible_by_axis)

    @staticmethod
    def dtype_check(input_dtypes, acceptable_dtypes):
        """compares ArrayType dtypes and returns a boolean to indicate
        compatability

        Args:
            input_dtype(tuple): ArrayType dtypes of the input data
            acceptable_dtypes(tuple): one of the acceptable input type array
                shapes

        Returns:
            compatible(bool): whether or the input dtype is compatible with
                this block
        """
        compatability_by_dtype = []

        # if they are the exact same, save time by returning early
        if input_dtypes == acceptable_dtypes:
            return True

        # if they aren't the exact same, check that all input types
        # are in the acceptable types
        for idtype in input_dtypes:
            compatability_by_dtype.append(idtype in acceptable_dtypes)

        return all(compatability_by_dtype)

    def output_given_input(self, input_types):
        """gets the given output(s) of this IoMap given a input type or types


        Args:
            input_types(input_type,tuple): input type or tuple of input types

        Returns:
            outputs(tuple): tuple of output types this block produces given the
                input_types

        Raises:
            IncompatibleTypes: if there is no compatible output associated
                with the input_types

        """
        outputs = set()

        if not isinstance(input_types, tuple):
            # make it a tuple if a single type is passed in
            input_types = (input_types,)

        for input_type in input_types:
            # if we have a non array type, then we can use a boolean comparison
            # to determine what outputs there should be
            if input_type in BLOCK_NON_ARRAY_TYPES:
                for i, o in self.non_arrays:
                    if input_type == i:
                        outputs.add(o)

            # if we have an array type, then we have to do a shape comparison
            elif isinstance(input_type, ArrayType):
                for i, o in self.arrays:
                    dtype_okay = self.dtype_check(input_type.dtypes, i.dtypes)
                    for input_shape in input_type.shapes:
                        shp_okay = self.shape_comparison(input_shape, i.shapes[0])
                        if (shp_okay and dtype_okay):
                            outputs.add(o)

        if len(outputs) > 0:
            return tuple(outputs)

        msg = "invalid input type, must be ({}) not {}".format(self.inputs,
                                                               input_type)
        raise IncompatibleTypes(msg)


class BaseBlock(object):
    """BaseBlock object which is the root class for SimpleBlock and BatchBlock
    subclasses

    This is the _building block_ (pun intended) for the entire imsciutils
    pipelining system. All Blocks, both Simple and Batch blocks, will inherit
    from this object. Which contains base functionality to setup a block's
    printers, unique name, standard input/output_shapes and special functions
    for pipeline objects to call

    Args:

        io_map(IoMap,dict): dictionary of input-output mappings for this
            Block
        name(str): name for this block, it will be automatically created/modified
            to make sure it is unique
        notes(str): a short description of this block
        requires_training(bool): whether or not this block will require
            training
        requires_labels(bool): whether or not this block will require
            labels during training

    Attributes:

        io_map(IoMap): object that maps inputs to this block to outputs
            subclass of tuple where I/O is stored as:
            ( (input1,output1),(input2,output2)... )
        name(str): unique name for this block
        notes(str): a short description of this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'

    """
    EXTANT = {}

    def __init__(self,
                 io_map,
                 name=None,
                 notes=None,
                 requires_training=False,
                 requires_labels=False,
                 ):
        # ----------- building a unique name for this block ------------
        if name is None:
            name = self.__class__.__name__

        # JM: keeping track of names in base class variable
        if name in self.EXTANT:
            self.EXTANT[name] += 1
        else:
            self.EXTANT[name] = 1
        name = name + '({})'.format( self.EXTANT[name] )


        # checking if notes were provided for this block
        if notes is None:
            notes = "No Description provided by the author"
        if not isinstance(notes,str):
            raise TypeError("notes must be a string description or None")

        self.io_map = IoMap(io_map)

        self.name = name
        self.notes = notes
        self.requires_training = requires_training
        self.requires_labels = requires_labels

        self.trained = False
        if not self.requires_training:
            self.trained = True

        self.printer = get_printer(self.name)

    def rename(self,name):
        assert isinstance(name,str),"name must be a string"
        self.name = name
        self.printer = get_printer(self.name)
        return self

    def train(self, data, labels=None):
        """(optional overload)trains the block if required

        users are expected to save pertinent variables as instance
        variables

        Args:
            data(list): list of datums to train on
            labels(list,None): corresponding label for each datum,
                None by default (for unsupervised systems)

        Returns:
            None
        """
        pass

    def before_process(self, data, labels=None):
        """(optional overload)function that runs before processing for
        optional functionality. this function takes in the full data list and
        label list. does nothing unless overloaded

        Args:
            data(list): list of datums to process
            labels(list,None): corresponding label for each datum,
                None by default (for unsupervised systems)
        """
        pass

    def after_process(self):
        """
        (optional overload)function that runs after processing for
        optional functionality. intended for optional use as a cleanup
        function

        Args:
            None
        """
        pass

    def _pipeline_train(self, data, labels=None):
        """function pipeline calls to train this block, modifies
        self.trained status

        Args:
            data(list): list of datums to train on
            labels(list,None): corresponding label for each datum,
                None by default (for unsupervised systems)

        Returns:
            None

        Raises:
            BlockRequiresLabels: if this block requires labels and None
                was passed in
        """
        if self.requires_labels and (labels is None):
            msg = "{} requires labels for training but none were passed in"\
                .format(self)
            raise BlockRequiresLabels(msg)

        if isinstance(labels,list):
            # if labels are passed in, we must have an equal number
            # of datums and labels
            if len(data) != len(labels):
                raise DataLabelMismatch(data, labels)

        self.train(data, labels)
        self.trained = True

    def _pipeline_process(self, data, labels=None):
        """function pipeline calls to process data using this block
        works with BatchBlocks and SimpleBlocks

        Args:
            data(list): list of datums to process
            labels(list,None): corresponding label for each datum,
                None by default (for unsupervised systems)

        Returns:
            processed(list): list of processed datums
            labels(list): list of corresponding labels for processed

        Raises:
            InvalidBlockInputData: if data is not a list or tuple
            InvalidProcessingStrategy: if processed output is not a list
            InvalidLabelStrategy: if labels output is not a list
            DataLabelMismatch: if there is mismatch in the number of labels
                and processed datums

        """
        # ---------------- INPUT ERROR CHECKING ------------------------
        if not isinstance(data, list):
            error_msg = "input data into a block must be a list"
            self.printer.error(error)
            raise InvalidBlockInputData(self)

        if labels is None:
            labels = [None] * len(data)

        if isinstance(labels,list):
            if len(data) != len(labels):
                raise DataLabelMismatch(processed, labels)

        # ----------- running block functions -----------------------
        # running prep function
        self.before_process(data, labels)

        # processing data and labels
        processed = self.process_strategy(data)
        labels = self.label_strategy(labels)

        # running post-process / cleanup function
        self.after_process()


        # --------------- OUTPUT ERROR CHECKING ------------------------
        # error checking for output types
        if not isinstance(processed, list):
            raise InvalidProcessStrategy(self)

        if not isinstance(labels, list):
            raise InvalidLabelStrategy(self)

        # making sure that we always have the same number of labels and datums
        if isinstance(labels,list):
            if len(processed) != len(labels):
                raise DataLabelMismatch(processed, labels)

        return processed, labels

    def process_strategy(self, data):
        """overarching processing management function for this block

        Args:
            data(list): list of datums to process

        Returns:
            processed (list): processed datums
        """
        raise NotImplementedError("'process_strategy' must be overloaded in all children")

    def label_strategy(self, labels):
        """overarching label management function for this block

        Args:
            labels(list,None): corresponding label for each datum,
                None by default (for unsupervised systems)

        Returns:
            labels (list): labels for datums (Nones for unsupervised systems)
        """
        raise NotImplementedError("'label_strategy' must be overloaded in all children")

    def __str__(self):
        return self.name

    def __repr__(self):
        return (str(self) + '\n' + self.notes)

    def prep_for_serialization(self):
        pass

    def restore_from_serialization(self):
        pass


class SimpleBlock(BaseBlock):
    """Block subclass that processes individual datums separately
    (as opposed to processing all data at once in a batch). This makes it useful
    for most CPU bound processing tasks as well as most functions in traditional
    computer vision that don't require an image sequence to process data

    Args:

        io_map(IoMap,dict): dictionary of input-output mappings for this
            Block
        name(str): name for this block, it will be automatically created/modified
            to make sure it is unique
        notes(str): a short description of this block
        requires_training(bool): whether or not this block will require
            training
        requires_labels(bool): whether or not this block will require
            labels during training

    Attributes:

        io_map(IoMap): object that maps inputs to this block to outputs
            subclass of tuple where I/O is stored as:
            ( (input1,output1),(input2,output2)... )
        name(str): unique name for this block
        notes(str): a short description of this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'

    """

    def process(self, datum):
        """(required overload)processes a single datum

        Args:
            datum: datum to process

        Returns:
            processed: datum processed by this block
        """
        raise NotImplementedError("'process' must be overloaded in all children")

    def label(self, lbl):
        """(optional overload)retrieves the label for this datum"""
        return lbl

    def process_strategy(self, data):
        """processes each datum using self.process and return list"""
        return [self.process(datum) for datum in data]

    def label_strategy(self, labels):
        """calls self.label for each datum and returns a list or Nonetype"""
        return [self.label(lbl) for lbl in labels]

    def __repr__(self):
        return (str(self) + '-(SimpleBlock)' + '\n' + self.notes)





class BatchBlock(BaseBlock):
    """Block subclass that processes datums as a batch
    (as opposed to processing each datum individually). This makes it useful
    for GPU accelerated tasks where processing data in batches frequently
    increases processing speed. It can also be used for algorithms that
    require working with a full image sequence.

    Args:

        io_map(IoMap,dict): dictionary of input-output mappings for this
            Block
        name(str): name for this block, it will be automatically created/modified
            to make sure it is unique
        notes(str): a short description of this block
        requires_training(bool): whether or not this block will require
            training
        requires_labels(bool): whether or not this block will require
            labels during training

    Attributes:

        io_map(IoMap): object that maps inputs to this block to outputs
            subclass of tuple where I/O is stored as:
            ( (input1,output1),(input2,output2)... )
        name(str): unique name for this block
        notes(str): a short description of this block
        requires_training(bool): whether or not this block will require
            training
        trained(bool): whether or not this block has been trained, True
            by default if requires_training = False
        printer(iu.Printer): printer object for this block,
            registered to 'name'

    """

    def batch_process(self, data):
        """(required overload)processes a list of data using this block's
        algorithm

        Args:
            data(list): list of datums to process

        Returns:
            process(list): list of processed datums
        """
        error_msg = "'batch_process' must be overloaded in all children"
        raise NotImplementedError(error_msg)

    def labels(self, labels):
        """(optional overload) returns all labels for input datums or None"""
        return labels

    def process_strategy(self, data):
        """runs self.batch_process"""
        return self.batch_process(data)

    def label_strategy(self, labels):
        """runs self.labels"""
        return self.labels(labels)

    def __repr__(self):
        return (str(self) + '-(BatchBlock)' + '\n' + self.notes)

# END
