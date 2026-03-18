import numpy as np

from nn.abstract import ParameterNode


class Embedding(ParameterNode):

    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.weights = np.random.randn(vocab_size, embedding_size)
        self.weights_grad = np.zeros_like(self.weights)

    def forward(self, input_idx: np.ndarray) -> np.ndarray:
        """
        Return embeddings corresponding to the input indices
        :param input_idx: array of shape (batch_size, ) with indices in range [0, vocab_size)
        :return: array of shape (embedding_size, batch_size)
        """
        return self.weights[input_idx, :]

    def _compute_input_grad(self, input_idx: np.ndarray, output_grad: np.ndarray):
        """
        Return a gradient of the transformation wrt the input
        :param input_idx: array of shape (batch_size, ) with indices in range [0, vocab_size)
        :param output_grad: array of shape (batch_size, embedding_size), gradient of a wrapping transformation wrt its input
        :return: array of shape (batch_size, vocab_size)
        """
        assert output_grad.shape[1] == self.embedding_size

        return output_grad @ self.weights.T[:, input_idx]

    def _update_parameters_grad(self, input_idx: np.ndarray, output_grad: np.ndarray):
        """
        Updates gradient of weights and biases of the current transformation
        :param input_idx: array of shape (batch_size, ) with indices in range [0, vocab_size)
        :param output_grad: array of shape (batch_size, embedding_size), gradient of a wrapping transformation wrt its input
        """
        np.add.at(self.weights_grad, input_idx, output_grad)

    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights)


class Linear(ParameterNode):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(1, output_size)

        self.weights_grad = np.zeros_like(self.weights)
        self.biases_grad = np.zeros_like(self.biases)

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Return transformed input
        :param input: array of shape (batch_size, input_size)
        :return: array of shape (batch_size, output_size)
        """
        return input @ self.weights + self.biases

    def _compute_input_grad(self, input: np.ndarray, output_grad: np.ndarray):
        """
        Return a gradient of the transformation wrt the input
        :param input: array of shape (batch_size, input_size)
        :param output_grad: array of shape (batch_size, output_size), gradient of a wrapping transformation wrt its input
        :return: array of shape (batch_size, input_size)
        """
        return output_grad @ self.weights.T

    def _update_parameters_grad(self, input: np.ndarray, output_grad: np.ndarray):
        """
        Updates gradient of weights and biases of the current transformation
        :param input: array of shape (batch_size, input_size)
        :param output_grad: array of shape (batch_size, output_size), gradient of a wrapping transformation wrt its input
        """
        self.weights += input.T @ output_grad
        self.biases += output_grad.sum(axis=0)

    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights)
        self.biases_grad = np.zeros_like(self.biases)

class Sequential(ParameterNode):

    def __init__(self, nodes: list[ParameterNode]):
        super().__init__()
        self.nodes = nodes

    def forward(self, input: np.ndarray) -> np.ndarray:
        node_input = input
        for node in self.nodes:
            node_input = node(node_input)
        return node_input

    def zero_grad(self):
        for node in self.nodes:
            node.zero_grad()

    def backward(self, input: np.ndarray, output_grad: np.ndarray):
        node_output_grad = output_grad
        idx = len(self.nodes) - 1
        while idx > 1:
            node_input = self.nodes[idx - 1].output
            node_output_grad = self.nodes[idx].backward(node_input, node_output_grad)
            idx -= 1
        return self.nodes[0].backward(input, node_output_grad)

    def _update_parameters_grad(self, input: np.ndarray, output_grad: np.ndarray):
        pass

    def _compute_input_grad(self, input: np.ndarray, output_grad: np.ndarray):
        pass