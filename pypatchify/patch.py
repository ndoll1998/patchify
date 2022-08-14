from typing import Optional, Sequence, Callable, Generic, TypeVar
from itertools import chain

T = TypeVar('T')

class Patchify(Generic[T]):

    # get shape and strides from tensor object
    shape:Callable
    strides:Callable
    # tensor operations
    reshape:Callable
    transpose:Callable
    as_strided:Callable

    @classmethod
    def sliding_window(cls, t, window_dims:Sequence[int], strides:Sequence[int]) -> T:
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
        # get window dim
        n = len(window_dims)
        # get tensor shape and strides
        t_shape = cls.shape(t)
        t_strides = cls.strides(t)
        # check dimensions
        if len(t_shape) < n:
            raise ValueError("Cannot create sliding windows of dimension %i on tensor of dimension %i." % (n, len(t_shape)))
        # compute new shape and strides for sliding window view
        shape = chain(t_shape[:-n], ((d - k) // s + 1 for d, k, s in zip(t_shape[-n:], window_dims, strides)), window_dims)
        strides = chain(t_strides[:-n], (ts * ws for ts, ws in zip(t_strides[-n:], strides)), t_strides[-n:])
        # apply shape and strides
        return cls.as_strided(t, tuple(shape), tuple(strides))

    @classmethod
    def patchify(cls, t, patch_sizes:Sequence[int]) -> T:
        """ Patchify n-dimension tensor with given patch size 

            Args:
                t (Tensor): 
                    tensor object of dimension k which to patch into non-overlapping windows
                patch_sizes (Sequence[int]): 
                    dimensions of patches. Patch dimensionality n must be <= k.
                    Patches will be applied to the last n dimensions of the input tensor.

            Returns:
                patches (Tensor): 
                    tensor containing patches in shape (..., p_1, ..., p_n, *patch_sizes)
                    where pi is the number of sliding windows in the i-th dimension
        """
        return cls.sliding_window(t, patch_sizes, patch_sizes)

    @classmethod
    def unpatchify(cls, t, unpatched_sizes:Sequence[int]) -> T:
        """ Merge patches of given patched tensor

            Args:
                t (Tensor):
                    tensor storing patches to merge. Shape of the tensor must match
                    (..., p_1, ..., p_n, *patch_sizes) where p_i is the
                    number of patches in the i-th dimension
                unpatched_sizes (Sequence[int]):
                    sizes of the unpatched images/volumes, i.e. 
                    unpatched_sizes[i-1] = p_i * patch_sizes[i]
        
            Returns:
                unpatched (Tensor):
                    tensor of shape (..., *unpatched_sizes)
                    storing the merged patches
        """
        # get tensor shape and patch dimensionality
        shape = cls.shape(t)
        k, n = len(shape), len(unpatched_sizes)
        # re-order dimensions
        dim_idx = chain(range(0, k - 2*n), *zip(range(k - 2*n, k-n), range(k-n, k)))
        t = cls.transpose(t, tuple(dim_idx))
        # collapse patches
        merged_shape = chain(shape[:k-2*n], unpatched_sizes)
        return cls.reshape(t, tuple(merged_shape))

    @classmethod
    def collapse_dims(cls, t, dims:Sequence[int], target_dim:int =0) -> T:
        """ Collapse multiple dimensions of a given tensor 

            Args:
                t (Tensor): input tensor
                dims (Sequence[int]): dimensions to collapse in the input tensor
                target_dim (int): dimension into which to collapse the given dimensions. Defaults to 0.
        
            Returns:
                collapsed (Tensor): tensor with dimensions collapsed into target dimension
        """
        # get tensor shape and dimensionality
        shape = cls.shape(t)
        n, k = len(shape), len(dims)
        # check dimensions to collapse
        if any(d > len(shape) for d in dims):
            raise ValueError("Dimension out of range in `collapse_dims`")
        # get remaining dimensions
        set_dims = set(dims) # faster lookup :)
        remain_dims = [i for i in range(n) if i not in set_dims]
        # transpose dimensions
        dim_idx = chain(remain_dims[:target_dim], dims, remain_dims[target_dim:])
        t = cls.transpose(t, tuple(dim_idx))
        # collapse dimensions
        shape = cls.shape(t) # get new shape
        shape = shape[:target_dim] + (-1,) + shape[target_dim + k:]
        return cls.reshape(t, shape)

    @classmethod
    def patchify_to_batches(cls, t, patch_sizes:Sequence[int], batch_dim:Optional[int] =0) -> T:
        """ Patchify n-dimension tensor with given patch size and collapse patching
            dimensions into batch dimension.
        
            Args:
                t (Tensor):
                    tensor object of dimension k which to patch into non-overlapping windows
                patch_sizes (Sequence[int]): 
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
        # get dimensions
        k = len(cls.shape(t))
        n = len(patch_sizes)
        # patchify and collapse
        return cls.collapse_dims(
            cls.patchify(t, patch_sizes),
            dims=[batch_dim] + list(range(k-n, k)),
            target_dim=batch_dim
        )

    @classmethod
    def unpatchify_from_batches(cls, t, unpatched_sizes:Sequence[int], batch_dim:Optional[int] =0) -> T:
        """ Merge patches of given patched tensor with patched collapsed
            into batch dimension

            Args:
                t (Tensor): tensor storing patches to merge.
                unpatched_sizes (Sequence[int]):
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
        # get shape
        shape = cls.shape(t)
        n, k = len(unpatched_sizes), len(shape)
        # get patch-sizes and number of patches in each dimension
        patch_sizes = shape[-n:]
        n_patches = [i // j for i, j in zip(unpatched_sizes, patch_sizes)]
        # split patches from batch
        patched_shape = chain(shape[:batch_dim], [-1], n_patches, shape[batch_dim+1:])
        t = cls.reshape(t, tuple(patched_shape))
        # re-organize dimensions
        dim_idx = chain(range(batch_dim+1), range(batch_dim+1+n, k), range(batch_dim+1, batch_dim+1+n), range(k, k+n))
        t = cls.transpose(t, tuple(dim_idx))
        # merge patches
        return cls.unpatchify(t, unpatched_sizes)

