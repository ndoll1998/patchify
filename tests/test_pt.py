import torch
from pypatchify.pt import pt

class TestPT:

    def test_patchify_unpatchify(self):
        
        for shape, patch in [
            [(4, 4), (2, 2)],
            [(8, 8, 8), (2, 2)],
            [(8, 3, 16, 16), (8, 8)],
            [(8, 4, 32, 32, 64), (2, 16, 4, 8)]
        ]:
            n = len(patch)
            t_orig = torch.rand(shape)
            t = pt.patchify(t_orig, patch)
            t = pt.unpatchify(t, shape[-n:])
            assert (t == t_orig).all()
        
    def test_batched_patchify_unpatchify(self):
        
        for shape, patch, bdim in [
            [(1, 4, 4), (2, 2), 0],
            [(8, 8, 8), (2, 2), 0],
            [(8, 3, 16, 16), (8, 8), 0],
            [(8, 3, 16, 16), (8, 8), 1],
            [(8, 4, 32, 32, 64), (2, 16, 4, 8), 0]
        ]:
            n = len(patch)
            t_orig = torch.rand(size=shape)
            t = pt.patchify_to_batches(t_orig, patch, bdim)
            t = pt.unpatchify_from_batches(t, shape[-n:], bdim)
            assert (t == t_orig).all()
        
