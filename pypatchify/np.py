import numpy
from .patch import Patchify

class np(Patchify[numpy.ndarray]):
    """ Collection of patchification functionality for numpy arrays """
    # get shape and strides from tensor object
    shape = lambda t: t.shape
    strides = lambda t: t.strides
    # tensor operations
    reshape = numpy.reshape
    transpose = numpy.transpose
    as_strided = numpy.lib.stride_tricks.as_strided

