import pytest
from numpy import dtype

from nn import Embedding, Linear, Sequential
import torch
import numpy as np

@pytest.mark.parametrize("vocab_size, embedding_size", [
    (10_000, 300),
    (1000, 1000),
    (20, 2000),
])
def test_embedding_forward(vocab_size: int, embedding_size: int):
    custom_embed = Embedding(vocab_size, embedding_size)
    torch_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
    torch_embed.weight.data = torch.tensor(custom_embed.weights)

    x1 = np.random.randint(0, vocab_size, (128, ))
    x2 = torch.tensor(x1)

    actual = custom_embed(x1)
    expected = torch_embed(x2)

    assert actual.dtype == dtype("float64")
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected.detach().numpy())

@pytest.mark.parametrize("vocab_size, embedding_size", [
    (10_000, 300),
    (1000, 1000),
    (5, 2000),
])
def test_embedding_backward(vocab_size: int, embedding_size: int):
    custom_embed = Embedding(vocab_size, embedding_size)
    torch_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
    torch_embed.weight.data = torch.tensor(custom_embed.weights)
    x1 = np.random.randint(0, vocab_size, (128, ))
    x2 = torch.tensor(x1)

    actual = custom_embed(x1)
    expected = torch_embed(x2)

    output_grad = np.random.randn(*actual.shape)
    expected.backward(torch.from_numpy(output_grad))
    custom_embed.backward(x1, output_grad)
    actual_weight_grad = custom_embed.weights_grad
    expected_weight_grad = torch_embed.weight.grad.numpy()

    assert actual_weight_grad.dtype == dtype("float64")
    assert actual_weight_grad.shape == expected_weight_grad.shape

    assert np.allclose(actual_weight_grad, expected_weight_grad)

def test_embedding_zero_grad():
    vocab_size = 1000
    custom_embed = Embedding(vocab_size, 10)
    x = np.random.randint(0, vocab_size, (128, ))

    actual = custom_embed(x)
    output_grad = np.random.randn(*actual.shape)
    custom_embed.backward(x, output_grad)

    assert (custom_embed.weights_grad != 0).sum() > 0
    custom_embed.zero_grad()
    assert (custom_embed.weights_grad == 0).all()

@pytest.mark.parametrize("input_dim, output_dim", [
    (1, 10),
    (10, 1),
    (5, 5),
    (10_000, 20_000)
])
def test_linear_forward(input_dim, output_dim):
    custom_linear = Linear(input_dim, output_dim)
    torch_linear = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
    torch_linear.weight.data = torch.tensor(custom_linear.weights)
    torch_linear.bias.data = torch.tensor(custom_linear.biases)

    x1 = np.random.rand(128, input_dim)
    x2 = torch.tensor(x1)

    actual = custom_linear(x1)
    expected = torch_linear(x2)

    assert actual.dtype == dtype("float64")
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected.detach().numpy())


@pytest.mark.parametrize("input_dim, output_dim", [
    (1, 10),
    (10, 1),
    (5, 5),
    (10_000, 20_000)
])
def test_linear_backward(input_dim, output_dim):
    custom_linear = Linear(input_dim, output_dim)
    torch_linear = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
    torch_linear.weight.data = torch.tensor(custom_linear.weights)
    torch_linear.bias.data = torch.tensor(custom_linear.biases)

    x1 = np.random.rand(128, input_dim)
    x2 = torch.tensor(x1, requires_grad=True)

    actual = custom_linear(x1)
    expected = torch_linear(x2)

    output_grad = np.random.randn(*actual.shape)
    expected.backward(torch.from_numpy(output_grad))
    actual_input_grad = custom_linear.backward(x1, output_grad)
    expected_input_grad = x2.grad.numpy()
    actual_weight_grad = custom_linear.weights_grad
    actual_bias_grad = custom_linear.biases_grad
    expected_weight_grad = torch_linear.weight.grad.numpy()
    expected_bias_grad = torch_linear.bias.grad.numpy()

    assert actual_weight_grad.dtype == dtype("float64")
    assert actual_weight_grad.shape == expected_weight_grad.shape
    assert np.allclose(actual_weight_grad, expected_weight_grad)

    assert actual_bias_grad.dtype == dtype("float64")
    assert actual_bias_grad.shape == expected_bias_grad.shape
    assert np.allclose(actual_bias_grad, expected_bias_grad)

    assert actual_input_grad.dtype == dtype("float64")
    assert actual_input_grad.shape == expected_input_grad.shape
    assert np.allclose(actual_input_grad, expected_input_grad)

def test_linear_zero_grad():
    input_dim = 1000
    custom_linear = Linear(input_dim, 10)
    x = np.random.randint(0, input_dim, (128, input_dim))

    actual = custom_linear(x)
    output_grad = np.random.randn(*actual.shape)
    custom_linear.backward(x, output_grad)

    assert (custom_linear.weights_grad != 0).sum() > 0
    custom_linear.zero_grad()
    assert (custom_linear.weights_grad == 0).all()