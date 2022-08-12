# PyPatchify

![Tests Master](https://github.com/ndoll1998/patchify/workflows/Tests/badge.svg)
![Publish PyPI](https://github.com/ndoll1998/patchify/workflows/PyPI/badge.svg)

Fast and easy image and n-dimensional volume patchification

## Install and Requirements

The package can be installed from [PyPI](https://pypi.org/project/pypatchify/):

```bash
pip install pypatchify
```

It supports both numpy arrays and pytorch tensors. However pytorch is not strictly required. The only necessary dependency is:

 - numpy >= 1.21.5

To install all dependencies (including pytorch and pytest) run the following

```bash
python -m pip install requirements.txt
```

## Hello World

The library is designed to be easy to use while keeping the computational overhead as low as possible. The following simple example shows how to patchify and unpatchify a batch of rgb images:

```python
import pypatchify
import numpy as np
# create 16 random rgb images
imgs = np.random.uniform(0, 1, size=(16, 3, 256, 256))
# patchify into non-overlapping blocks of size 64x64
patched_imgs = pypatchify.patchify(imgs, (64, 64))

# re-create the original images from the patches
imgs = pypatchify.unpatchify(patched_imgs, (256, 256))
```

In case the created patches are further processed, for example by passing them through a neural network, it might make sense to collapse the different patches into a batch size as follows:

```python
imgs = np.random.uniform(0, 1, size=(16, 3, 256, 256))
# patchify into non-overlapping blocks of size 64x64
patched_imgs = pypatchify.patchify_to_batches(imgs, (64, 64), batch_dim=0)
# re-create the original images from the patches
imgs = pypatchify.unpatchify_from_batches(patched_imgs, (256, 256), batch_dim=0)
```

Note that the implementations are not restricted to 2d images only but can patchify and unpatchify any multi-dimensional volume:

```python
vols = np.random.uniform(0, 1, size=(16, 32, 32, 64, 64))
# patchify into non-overlapping blocks of size 64x64
patched_vols = pypatchify.patchify_to_batches(vols, (16, 8, 32, 16), batch_dim=0)
# re-create the original images from the patches
vols = pypatchify.unpatchify_from_batches(patched_vols, (32, 32, 64, 64), batch_dim=0)
```

## GPU-Acceleration

Also when working with neural networks its probably more convenient to directly work with pytorch tensors. This can be done by simply passing the torch tensors to the function at hand. Note that all implementations allow gpu-tensors which drastically decrease the runtime of any of the patchification functions. Also there is no need to move memory between cpu and gpu.

```python
import torch
import pypatchify
# create a random img tensor and move to cuda
imgs = torch.rand((16, 3, 256, 256)).cuda()
# patchify into non-overlapping blocks of size 64x64
patched_imgs = pypatchify.patchify_to_batches(imgs, (64, 64), batch_dim=0)
# re-create the original images from the patches
imgs = pypatchify.unpatchify_from_batches(patched_imgs, (256, 256), batch_dim=0)
```

## Other Frameworks

The library makes it very easy to support other frameworks besides numpy and pytorch. All work that needs to be done is to implement the following few functions:

 - shape: get the shape of a given tensor
 - strides: get the strides of the underlying memory
 - reshape: reshape a given tensor to a given shape
 - transpose: permute the dimensions of a given tensor to a given permuation
 - as_strided: apply a given shape and strides to the memory of a given tensor

Note that most frameworks already support these functions. To now integrate the framework just inherit from the `pypatchify.patch.Patchify` class and enter the functions:

```python
class NewFramework(Patchify):
    # get shape and strides from tensor object
    shape:Callable
    strides:Callable
    # tensor operations
    reshape:Callable
    transpose:Callable
    as_strided:Callable
```

The class now holds static member functions for all the patchification functionality including `patchify`, `unpatchify`, etc.
