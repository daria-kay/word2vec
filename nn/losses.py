import numpy as np
from scipy.special import log_expit, expit

from .abstract import Node


class BinaryCrossEntropy(Node):

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate binary classification loss
        :param logits: array of shape (batch_size, )
        :param labels: array of shape (batch_size, ), OHE vector of positive samples
        :return:
        """
        mask = labels.astype(bool)
        return -(np.where(mask, log_expit(logits), 0) + np.where(~mask, log_expit(-logits), 0))


    def backward(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculate a gradient of the loss funciton wrt to the input
        :param logits: array of shape (batch_size, )
        :param labels: array of correct class labels (batch_size, )
        :return: array of shape (batch_size, )
        """
        return expit(logits) - labels