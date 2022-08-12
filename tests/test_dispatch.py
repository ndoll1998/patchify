import numpy
import torch
import pypatchify

class TestDispatch:

    def test_numpy(self):

        t = numpy.random.uniform(0, 1, size=(4, 4))
        t = pypatchify.patchify(t, (2,))
        assert isinstance(t, numpy.ndarray)
        t = pypatchify.unpatchify(t, (4,))
        assert isinstance(t, numpy.ndarray)
    
    def test_torch(self):

        t = torch.rand(4, 4)
        t = pypatchify.patchify(t, (2,))
        assert isinstance(t, torch.Tensor)
        t = pypatchify.unpatchify(t, (4,))
        assert isinstance(t, torch.Tensor)
