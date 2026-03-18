import numpy as np
import scipy.special

from .abstract import Node

class NegativeLogLikelihood(Node):

    def forward(self, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate cross entropy loss for the given logits
        :param logits: array of shape (n_samples, vocab_size)
        :param target: array of correct class labels (n_samples, n_classes)
        :return: array of shape (n_samples, 1)
        """
        x_idx = np.arange(logits.shape[0]).reshape(-1, 1)
        return -scipy.special.log_softmax(logits, axis=1)[x_idx, target].sum(axis=1, keepdims=True)


    def backward(self, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate a gradient of cross entropy loss wrt to the input
        :param logits: array of shape (n_samples, vocab_size)
        :param target: array of correct class labels (n_samples, n_classes)
        :return: array of shape (n_samples, vocab_size)
        """
        input_softmax = scipy.special.softmax(logits, axis=1)
        window_elements_term = np.zeros_like(logits)
        x_idx = np.arange(logits.shape[0]).reshape(-1, 1)
        np.add.at(window_elements_term, (x_idx, target), 1)
        window_size = target.shape[1]
        return window_size * input_softmax - window_elements_term