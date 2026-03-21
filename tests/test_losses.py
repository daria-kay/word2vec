import numpy as np
import pytest
import torch

from nn import NegativeLogLikelihood
from utils import assert_is_close


@pytest.mark.parametrize("vocab_size, window_size", [(100, 2), (10_000, 4), (10, 16), (10, 1)])
def test_nll_forward(vocab_size, window_size):
    batch_size = 128
    logits = np.random.randn(batch_size, vocab_size)
    torch_logits = torch.tensor(logits, requires_grad=True)
    target = np.random.randint(0, vocab_size, size=(batch_size, window_size))
    custom_criterion = NegativeLogLikelihood()

    actual_loss = custom_criterion(logits, target)
    expected_loss = calculate_torch_loss(torch_logits, target)

    assert_is_close(actual_loss, expected_loss)

@pytest.mark.parametrize("vocab_size, window_size", [(100, 2), (10_000, 4), (10, 16), (10, 1)])
def test_nll_backward(vocab_size, window_size):
    batch_size = 128
    logits = np.random.randn(batch_size, vocab_size)
    torch_logits = torch.tensor(logits, requires_grad=True)
    target = np.random.randint(0, vocab_size, size=(batch_size, window_size))
    custom_criterion = NegativeLogLikelihood()

    expected_loss = calculate_torch_loss(torch_logits, target)
    expected_loss.backward(torch.ones_like(expected_loss))
    actual_grad = custom_criterion.backward(logits, target)

    assert_is_close(actual_grad, torch_logits.grad)


def calculate_torch_loss(logits: torch.Tensor, target: np.ndarray):
    return -torch.nn.functional.log_softmax(logits, dim=1)[torch.arange(128).unsqueeze(1), target].sum(
        dim=1, keepdim=True)
