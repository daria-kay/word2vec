import numpy as np

from nn.abstract import ParameterNode


class Embedding(ParameterNode):

    def __init__(self, vocab_size: int, embedding_size: int):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.weights = np.random.uniform(
            low=-0.5 / self.embedding_size,
            high=0.5 / self.embedding_size,
            size=(vocab_size, embedding_size)
        )
        self.weights_grad = np.zeros_like(self.weights)

        super().__init__(
            [self.weights],
            [self.weights_grad],
            parameter_names=["embedding_weights"]
        )

    def forward(self, input_idx: np.ndarray) -> np.ndarray:
        """
        Return embeddings corresponding to the input indices
        :param input_idx: array of shape (batch_size, ) with indices in range [0, vocab_size)
        :return: array of shape (batch_size, embedding_size)
        """
        return self.weights[input_idx, :]

    def _compute_input_grad(self, input_idx: np.ndarray, output_grad: np.ndarray):
        pass

    def _update_parameters_grad(self, input_idx: np.ndarray, output_grad: np.ndarray):
        """
        :param input_idx: array of shape (batch_size, ) with indices in range [0, vocab_size)
        :param output_grad: array of shape (batch_size, embedding_size), gradient of a wrapping transformation wrt its input
        """
        np.add.at(self.weights_grad, input_idx, output_grad)

    def zero_grad(self):
        self.weights_grad.fill(0.0)


class SkipGram(ParameterNode):

    def __init__(self, vocab_size: int, embedding_size: int):
        self.central_embeddings = Embedding(vocab_size=vocab_size, embedding_size=embedding_size)
        self.context_embeddings = Embedding(vocab_size=vocab_size, embedding_size=embedding_size)
        params = self.central_embeddings.parameters + self.context_embeddings.parameters
        param_grads = self.central_embeddings.parameter_grads + self.context_embeddings.parameter_grads
        super().__init__(parameters=params, parameter_grads=param_grads, parameter_names=['center', 'context'])

    def forward(self, central_idx: np.ndarray, context_idx: np.ndarray) -> np.ndarray:
        """
        Return dot product for each central words and the corresponding context words
        :param central_idx: array of shape (batch_size, ) with indices in range [0, vocab_size)
        :param context_idx: array of shape (batch_size, ) with indices for context words
        :return: array of shape (batch_size, )
        """
        central_words = self.central_embeddings(central_idx)  # (batch_size, embedding_size)
        context_words = self.context_embeddings(context_idx)  # (batch_size, embedding_size)
        return np.sum(central_words * context_words, axis=1)  # (batch_size, )

    def zero_grad(self):
        self.central_embeddings.zero_grad()
        self.context_embeddings.zero_grad()

    def _update_parameters_grad(self, central_idx: np.ndarray, context_idx: np.ndarray, output_grad: np.ndarray):
        """
        Perform backward pass
        :param central_idx: array of shape (batch_size, ) with indices for central words
        :param context_idx: array of shape (batch_size, ) with indices for context words
        :param output_grad: array of shape (batch_size, ), gradient of a wrapping transformation wrt its input
        :return:
        """
        grad_wrt_central = output_grad[:, None] * self.context_embeddings.output
        self.central_embeddings.backward(central_idx, grad_wrt_central)

        grad_wrt_context = output_grad[:, None] * self.central_embeddings.output
        self.context_embeddings.backward(context_idx, grad_wrt_context)

    def _compute_input_grad(self, central_idx: np.ndarray, context_idx: np.ndarray, output_grad: np.ndarray):
        pass