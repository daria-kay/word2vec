import numpy as np

from nn import Optimizer, ParameterNode


class Adam(Optimizer):

    def __init__(self, model: ParameterNode, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        super().__init__(model)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t = 0

        self._param_count = len(self.model.parameters)
        self._ms = [np.zeros_like(param_grad) for param_grad in self.model.parameter_grads]
        self._gs = [np.zeros_like(param_grad) for param_grad in self.model.parameter_grads]

    def step(self):
        self._t += 1
        for i in range(self._param_count):
            grad = self.model.parameter_grads[i]

            self._ms[i] = self.beta1 * self._ms[i] + (1 - self.beta1) * grad
            self._gs[i] = self.beta2 * self._gs[i] + (1 - self.beta2) * grad * grad
            momentum_corr = self._ms[i] / (1 - self.beta1 ** self._t)
            second_momentum_corr = self._gs[i] / (1 - self.beta2 ** self._t)

            param_delta = self.lr * momentum_corr / (np.sqrt(second_momentum_corr) + self.eps)

            self.model.parameters[i] -= param_delta
