import numpy as np
import torch

def assert_is_close(actual: np.ndarray, expected: torch.Tensor):
    assert actual.dtype == np.dtype("float64")
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected.detach().numpy())