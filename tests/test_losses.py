import numpy as np
import torch

from nn import BinaryCrossEntropy
from utils import assert_is_close

BATCH_SIZE = 100


def test_bce_forward():
    logits = np.random.randn(BATCH_SIZE)
    positive_idx = np.random.choice(np.arange(0, BATCH_SIZE), size=(int(BATCH_SIZE * 0.5),), replace=False)
    target = np.zeros_like(logits)
    target[positive_idx] = 1
    custom_criterion = BinaryCrossEntropy()

    actual_loss = custom_criterion(logits, target)
    expected_loss, _ = calculate_torch_loss(logits, target)

    assert_is_close(actual_loss, expected_loss)


def test_bce_backward():
    logits = np.random.randn(BATCH_SIZE)
    positive_idx = np.random.choice(np.arange(0, BATCH_SIZE), size=(int(BATCH_SIZE * 0.5),), replace=False)
    target = np.zeros_like(logits)
    target[positive_idx] = 1
    custom_criterion = BinaryCrossEntropy()

    expected_loss, torch_logits = calculate_torch_loss(logits, target)
    expected_loss.backward(torch.ones_like(expected_loss))
    actual_grad = custom_criterion.backward(logits, target)

    assert_is_close(actual_grad, torch_logits.grad)


def calculate_torch_loss(logits: np.ndarray, target: np.ndarray):
    logits = torch.from_numpy(logits)
    logits.requires_grad_(True)
    target = torch.from_numpy(target)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none'), logits
