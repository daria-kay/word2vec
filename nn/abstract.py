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
    def backward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplemented

class ParameterNode(Node, ABC):

    @abstractmethod
    def __init__(self, parameters, parameter_grads):
        super().__init__()
        self.parameters = parameters
        self.parameter_grads = parameter_grads

    def backward(self,  *args, **kwargs) -> np.ndarray:
        input_grad = self._compute_input_grad( *args, **kwargs)
        self._update_parameters_grad( *args, **kwargs)
        return input_grad

    @abstractmethod
    def zero_grad(self):
        raise NotImplemented

    @abstractmethod
    def _update_parameters_grad(self,  *args, **kwargs):
        raise NotImplemented

    @abstractmethod
    def _compute_input_grad(self,  *args, **kwargs):
        raise NotImplemented

class Optimizer(ABC):

    def __init__(self, model: ParameterNode):
        self.model = model

    @abstractmethod
    def step(self):
        raise NotImplemented

    def zero_grad(self):
        self.model.zero_grad()







