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
    x1 = np.random.randint(0, vocab_size, (128, ))
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
    x = np.random.randint(0, vocab_size, (128, ))

    actual = custom_embed(x)
    output_grad = np.random.randn(*actual.shape)
    custom_embed.backward(x, output_grad)

    assert np.any(custom_embed.weights_grad)
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

    assert_is_close(actual, expected)


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

    assert_is_close(custom_linear.weights_grad, torch_linear.weight.grad)
    assert_is_close(custom_linear.biases_grad, torch_linear.bias.grad)
    assert_is_close(actual_input_grad, x2.grad)

def test_linear_zero_grad():
    input_dim = 1000
    custom_linear = Linear(input_dim, 10)
    x = np.random.randint(0, input_dim, (128, input_dim))

    actual = custom_linear(x)
    output_grad = np.random.randn(*actual.shape)
    custom_linear.backward(x, output_grad)

    assert np.any(custom_linear.weights_grad)
    custom_linear.zero_grad()
    assert (custom_linear.weights_grad == 0).all()


@pytest.mark.parametrize("embedding_dim, vocab_size", [
    (10, 1000),
    (300, 10_000),
    (100, 10),
])
def test_sequential_forward(embedding_dim, vocab_size):
    custom_embed = Embedding(vocab_size, embedding_dim)
    torch_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    torch_embed.weight.data = torch.tensor(custom_embed.weights)

    custom_linear = Linear(embedding_dim, vocab_size)
    torch_linear = torch.nn.Linear(in_features=embedding_dim, out_features=vocab_size)
    torch_linear.weight.data = torch.tensor(custom_linear.weights)
    torch_linear.bias.data = torch.tensor(custom_linear.biases)

    sequential_custom = Sequential([custom_embed, custom_linear])
    sequential_torch = torch.nn.Sequential(torch_embed, torch_linear)

    x1 = np.random.randint(0, vocab_size, size=(128,))
    x2 = torch.tensor(x1)
    actual_y = sequential_custom(x1)
    expected_y = sequential_torch(x2)

    assert_is_close(actual_y, expected_y)


@pytest.mark.parametrize("embedding_dim, vocab_size", [
    (10, 1000),
    (300, 10_000),
    (100, 10),
])
def test_sequential_backward(embedding_dim, vocab_size):
    custom_embed = Embedding(vocab_size, embedding_dim)
    torch_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    torch_embed.weight.data = torch.tensor(custom_embed.weights)

    custom_linear = Linear(embedding_dim, vocab_size)
    torch_linear = torch.nn.Linear(in_features=embedding_dim, out_features=vocab_size)
    torch_linear.weight.data = torch.tensor(custom_linear.weights)
    torch_linear.bias.data = torch.tensor(custom_linear.biases)

    sequential_custom = Sequential([custom_embed, custom_linear])
    sequential_torch = torch.nn.Sequential(torch_embed, torch_linear)

    x1 = np.random.randint(0, vocab_size, size=(128,))
    x2 = torch.tensor(x1)
    actual_y = sequential_custom(x1)
    expected_y = sequential_torch(x2)

    output_grad = np.random.randn(*actual_y.shape)
    expected_y.backward(torch.tensor(output_grad))
    sequential_custom.backward(x1, output_grad)

    assert_is_close(custom_linear.weights_grad, torch_linear.weight.grad)
    assert_is_close(custom_linear.biases_grad, torch_linear.bias.grad)
    assert_is_close(custom_embed.weights_grad, torch_embed.weight.grad)

def test_sequential_zero_grad():
    vocab_size = 100
    embedding_dim = 2

    custom_embed = Embedding(vocab_size, embedding_dim)
    custom_linear = Linear(embedding_dim, vocab_size)
    sequential_custom = Sequential([custom_embed, custom_linear])

    x = np.random.randint(0, vocab_size, size=(128,))
    y = sequential_custom(x)

    output_grad = np.random.randn(*y.shape)
    sequential_custom.backward(x, y)

    assert np.any(custom_linear.weights_grad)
    assert np.any(custom_linear.biases_grad)
    assert np.any(custom_embed.weights_grad)

    sequential_custom.zero_grad()

    assert np.all(custom_linear.weights_grad == 0)
    assert np.all(custom_linear.biases_grad == 0)
    assert np.all(custom_embed.weights_grad == 0)


def assert_is_close(actual: np.ndarray, expected: torch.Tensor):
    assert actual.dtype == dtype("float64")
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected.detach().numpy())