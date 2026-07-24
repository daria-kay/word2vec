from abc import ABC, abstractmethod

import numpy as np


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

    def __init__(self, parameters: list, parameter_grads: list, parameter_names: list[str] = None):
        super().__init__()
        self.parameters = parameters
        self.parameter_grads = parameter_grads
        if not parameter_names:
            self.parameter_names = []
        else:
            self.parameter_names = parameter_names

    def backward(self, *args, **kwargs) -> np.ndarray:
        input_grad = self._compute_input_grad(*args, **kwargs)
        self._update_parameters_grad(*args, **kwargs)
        return input_grad

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def _update_parameters_grad(self, *args, **kwargs):
        """
        Updates the gradient of the transformation wrt its parameters
        """
        pass

    @abstractmethod
    def _compute_input_grad(self, *args, **kwargs) -> np.ndarray:
        """
        Return the gradient of the transformation wrt its input
        :param input: array of shape (batch_size, input_size)
        :param output_grad: array of shape (batch_size, output_size), gradient of a wrapping transformation wrt its input
        :return: the gradient value of the transformation
        """
        pass


class Optimizer(ABC):

    def __init__(self, model: ParameterNode):
        self.model = model
        self._param_count = len(self.model.parameters)
        self.parameter_norms = None
        self.gradient_norms = None

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        self.model.zero_grad()

    def _save_step_info(self, param_id, param, param_grad):
        if self.gradient_norms is None:
            self.parameter_norms = dict()
            self.gradient_norms = dict()

        param_name = param_id
        if self.model.parameter_names is not None:
            param_name = self.model.parameter_names[param_id]

        if param_name not in self.gradient_norms:
            self.parameter_norms[param_name] = []
            self.gradient_norms[param_name] = []

        self.parameter_norms[param_name].append(np.linalg.norm(param))
        self.gradient_norms[param_name].append(np.linalg.norm(param_grad))