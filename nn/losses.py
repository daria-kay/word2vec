import numpy as np
from scipy.special import softmax, log_softmax, log_expit

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
        return -log_softmax(logits, axis=1)[x_idx, target].sum(axis=1, keepdims=True)


    def backward(self, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate a gradient of cross entropy loss wrt to the input
        :param logits: array of shape (n_samples, vocab_size)
        :param target: array of correct class labels (n_samples, n_classes)
        :return: array of shape (n_samples, vocab_size)
        """
        input_softmax = softmax(logits, axis=1)
        window_elements_term = np.zeros_like(logits)
        x_idx = np.arange(logits.shape[0]).reshape(-1, 1)
        np.add.at(window_elements_term, (x_idx, target), 1)
        window_size = target.shape[1]
        return window_size * input_softmax - window_elements_term


class NegativeSamplingLoss(Node):

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate binary classification loss for k - 1 negative samples and one positive
        :param logits: array of shape (n_samples, k)
        :param labels: array of shape (n_samples, k), OHE vectors of positive class
        :return:
        """
        mask = labels.astype(bool)
        return -(np.where(mask, log_expit(logits), 0).sum(axis=1) +  np.where(~mask, log_expit(-logits), 0).sum(axis=1))


    def backward(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate a gradient of the loss funciton wrt to the input
        :param logits: array of shape (n_samples, vocab_size)
        :param target: array of correct class labels (n_samples, n_classes)
        :return: array of shape (n_samples, vocab_size)
        """
        return log_expit(logits) - labels