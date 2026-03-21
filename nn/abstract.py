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
    def __init__(self, parameters: list, parameter_grads: list, parameter_names: list[str] = None):
        super().__init__()
        self.parameters = parameters
        self.parameter_grads = parameter_grads
        if not parameter_names:
            self.parameter_names = []
        else:
            self.parameter_names = parameter_names

    def backward(self,  *args, **kwargs) -> np.ndarray:
        input_grad = self._compute_input_grad(*args, **kwargs)
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
        self.gradient_norms = dict()

    @abstractmethod
    def step(self):
        raise NotImplemented

    def zero_grad(self):
        self.model.zero_grad()







