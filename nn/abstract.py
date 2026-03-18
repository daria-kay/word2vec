import numpy as np
from abc import ABC, abstractmethod

class Node(ABC):

    def __init__(self):
        self.output = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        self.output = self.forward(*args, **kwargs)
        return self.output

    @abstractmethod
    def backward(self, input: np.ndarray, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplemented

class ParameterNode(Node, ABC):

    def __init__(self):
        super().__init__()

    def backward(self, input: np.ndarray, output_grad: np.ndarray) -> np.ndarray:
        input_grad = self._compute_input_grad(input, output_grad)
        self._update_parameters_grad(input, output_grad)
        return input_grad

    @abstractmethod
    def zero_grad(self):
        raise NotImplemented

    @abstractmethod
    def _update_parameters_grad(self, input: np.ndarray, output_grad: np.ndarray):
        raise NotImplemented

    @abstractmethod
    def _compute_input_grad(self, input: np.ndarray, output_grad: np.ndarray):
        raise NotImplemented







