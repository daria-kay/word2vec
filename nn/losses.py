import numpy as np

from abstract import Node

class CrossEntropyLoss(Node):

    def forward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    def backward(self, input: np.ndarray, output_grad: np.ndarray) -> np.ndarray:
        pass