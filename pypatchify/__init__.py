from .np import np
from numpy import ndarray

try:
    import torch
    IS_TORCH_AVAILABLE = True
except ImportError:
    IS_TORCH_AVAILABLE = False

if IS_TORCH_AVAILABLE:
    from .pt import pt

from typing import Any, Optional, Sequence, Union

def patchify(t:Union[ndarray, Any], patch_sizes:Sequence[int]) -> Union[ndarray, Any]:
    """ Patchify n-dimension tensor with given patch size 

        Args:
            t (Tensor): 
                tensor object of dimension k which to patch into non-overlapping windows
            patch_sizes (Tuple[int]): 
                dimensions of patches. Patch dimensionality n must be <= k.
                Patches will be applied to the last n dimensions of the input tensor.

        Returns:
            patches (Tensor): 
                tensor containing patches in shape (..., p_1, ..., p_n, *patch_sizes)
                where pi is the number of sliding windows in the i-th dimension
    """
    # use pytorch if given
    if IS_TORCH_AVAILABLE and isinstance(t, torch.Tensor):
        return pt.patchify(t, patch_sizes)
    # fallback to numpy
    return np.patchify(t, patch_sizes)

    
def unpatchify(t:Union[ndarray, Any], unpatched_sizes:Sequence[int]) -> Union[ndarray, Any]:
    """ Merge patches of given patched tensor

        Args:
            t (Tensor):
                tensor storing patches to merge. Shape of the tensor must match
                (..., p_1, ..., p_n, *patch_sizes) where p_i is the
                number of patches in the i-th dimension
            unpatched_sizes (Tuple[int]):
                sizes of the unpatched images/volumes, i.e. 
                unpatched_sizes[i-1] = p_i * patch_sizes[i]
    
        Returns:
            unpatched (Tensor):
                tensor of shape (..., *unpatched_sizes)
                storing the merged patches
    """
    # use pytorch if given
    if IS_TORCH_AVAILABLE and isinstance(t, torch.Tensor):
        return pt.unpatchify(t, unpatched_sizes)
    # fallback to numpy
    return np.unpatchify(t, unpatched_sizes)
    
def patchify_to_batches(t:Union[ndarray, Any], patch_sizes:Sequence[int], batch_dim:Optional[int] =0) -> Union[ndarray, Any]:
    """ Patchify n-dimension tensor with given patch size and collapse patching
        dimensions into batch dimension.
    
        Args:
            t (Tensor):
                tensor object of dimension k which to patch into non-overlapping windows
            patch_sizes (Tuple[int]): 
                dimensions of patches. Patch dimensionality n must be <= k.
                Patches will be applied to the last n dimensions of the input tensor.
            batch_dim (Optional[int]):
                dimension in which to collapse the patching dimensions.
                Defaults to 0.

        Returns:
            patched (Tensor):
                patched tensor of shape S=(..., *patch_sizes) and S_{batch_dim} = b * p_1 * ... * p_n
                where b is the previous batch size and p_i is the number of patches in the i-th dimension
    """
    # use pytorch if given
    if IS_TORCH_AVAILABLE and isinstance(t, torch.Tensor):
        return pt.patchify_to_batches(t, patch_sizes, batch_dim)
    # fallback to numpy
    return np.patchify_to_batches(t, patch_sizes, batch_dim)
    
def unpatchify_from_batches(t:Union[ndarray, Any], unpatched_sizes:Sequence[int], batch_dim:Optional[int] =0) -> Union[ndarray, Any]:
    """ Merge patches of given patched tensor with patched collapsed
        into batch dimension

        Args:
            t (Tensor): tensor storing patches to merge.
            unpatched_sizes (Tuple[int]):
                sizes of the unpatched images/volumes, i.e. 
                unpatched_sizes[i-1] = p_i * patch_sizes[i]
            batch_dim (Optional[int]):
                dimension in which the batch and patches are collapsed.
                Defaults to 0.

        Returns:
            unpatched (Tensor):
                tensor of shape (..., *unpatched_sizes)
                storing the merged patches
    """
    # use pytorch if given
    if IS_TORCH_AVAILABLE and isinstance(t, torch.Tensor):
        return pt.unpatchify_from_batches(t, unpatched_sizes, batch_dim)
    # fallback to numpy
    return np.unpatchify_from_batches(t, unpatched_sizes, batch_dim)

def collapse_dims(cls, t:Union[ndarray, Any], dims:Sequence[int], target_dim:int =0) -> Union[ndarray, Any]:
    """ Collapse multiple dimensions of a given tensor 

        Args:
            t (Tensor): input tensor
            dims (Sequence[int]): dimensions to collapse in the input tensor
            target_dim (int): dimension into which to collapse the given dimensions. Defaults to 0.
    
        Returns:
            collapsed (Tensor): tensor with dimensions collapsed into target dimension
    """
    # use pytorch if given
    if IS_TORCH_AVAILABLE and isinstance(t, torch.Tensor):
        return pt.collapse_dims(t, dims, target_dim)
    # fallback to numpy
    return np.collapse_dims(t, dims, target_dim)

def sliding_window(cls, t:Union[ndarray, Any], window_dims:Sequence[int], strides:Sequence[int]) -> Union[ndarray, Any]:
    """ Sliding Window view on given tensor
        
        Args:
            t (Tensor): 
                tensor object of dimension k on which to apply the sliding window view
            window_dims (Sequence[int]): 
                dimensions of sliding window. Window dimensionality n must be <= k.
                Windows will be applied to the last n dimensions of the input tensor.
            strides (Sequence[int]): 
                strides (i.e. step size) of sliding window in each dimension

        Returns:
            windows (Tensor): 
                tensor containing sliding windows in shape (d_1, ..., d_(k-n), p_1, ..., p_n, *window_dims)
                where pi is the number of sliding windows in the i-th dimension
    """
    # use pytorch if given
    if IS_TORCH_AVAILABLE and isinstance(t, torch.Tensor):
        return pt.sliding_window(t, window_dims, strides)
    # fallback to numpy
    return np.sliding_window(t, window_dims, strides)
