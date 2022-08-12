import torch
from .patch import Patchify

class pt(Patchify):
    """ Collection of patchification functionality for torch tensors """
    # get shape and strides from tensor object
    shape = torch.Tensor.size
    strides = torch.Tensor.stride
    # tensor operations
    reshape = torch.reshape
    transpose = torch.permute
    as_strided = torch.as_strided
