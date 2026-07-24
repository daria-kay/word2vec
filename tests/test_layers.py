import numpy as np
import pytest
import torch

from nn import Embedding
from utils import assert_is_close


@pytest.mark.parametrize("vocab_size, embedding_size", [
    (10_000, 300),
    (1000, 1000),
    (20, 2000),
])
def test_embedding_forward(vocab_size: int, embedding_size: int):
    custom_embed = Embedding(vocab_size, embedding_size)
    torch_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
    torch_embed.weight.data = torch.tensor(custom_embed.weights)

    x1 = np.random.randint(0, vocab_size, (128,))
    x2 = torch.tensor(x1)

    actual = custom_embed(x1)
    expected = torch_embed(x2)

    assert_is_close(actual, expected)


@pytest.mark.parametrize("vocab_size, embedding_size", [
    (10_000, 300),
    (1000, 1000),
    (5, 2000),
])
def test_embedding_backward(vocab_size: int, embedding_size: int):
    custom_embed = Embedding(vocab_size, embedding_size)
    torch_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
    torch_embed.weight.data = torch.tensor(custom_embed.weights)
    x1 = np.random.randint(0, vocab_size, (128,))
    x2 = torch.tensor(x1)

    actual = custom_embed(x1)
    expected = torch_embed(x2)

    output_grad = np.random.randn(*actual.shape)
    expected.backward(torch.from_numpy(output_grad))
    custom_embed.backward(x1, output_grad)
    actual_weight_grad = custom_embed.weights_grad
    expected_weight_grad = torch_embed.weight.grad

    assert_is_close(actual_weight_grad, expected_weight_grad)


def test_embedding_zero_grad():
    vocab_size = 1000
    custom_embed = Embedding(vocab_size, 10)
    x = np.random.randint(0, vocab_size, (128,))

    actual = custom_embed(x)
    output_grad = np.random.randn(*actual.shape)
    custom_embed.backward(x, output_grad)

    assert np.any(custom_embed.weights_grad)
    custom_embed.zero_grad()
    assert (custom_embed.weights_grad == 0).all()

