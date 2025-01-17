"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))

        if bias:
          self.bias = Parameter(init.kaiming_uniform(out_features,1,device=device, dtype=dtype).reshape((1,out_features)))

        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias:
          y+=self.bias.broadcast_to((y.shape))
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        shape_prod = np.prod(shape)
        return ops.reshape(X, (X.shape[0], shape_prod // X.shape[0]))
        # return ops.reshape(X,(X.shape[0],-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y= ops.relu(x)
        return y
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for f in self.modules:
          x = f(x)

        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        exp = ops.logsumexp(logits, axes=1)

        y_onehot = init.one_hot(logits.shape[1], y,device=logits.device, dtype=logits.dtype)


        
        
        z_y = ops.summation(ops.multiply(logits, y_onehot),axes = 1)
        
        
        
        return ops.divide_scalar(ops.summation(exp-z_y),logits.shape[0])
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim,device = device),requires_grad=True)
        self.bias = Parameter(init.zeros(self.dim,device = device),requires_grad=True)
        self.running_mean = init.zeros(self.dim,device = device)
        self.running_var = init.ones(self.dim,device = device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]

        if self.training:
            ### Compute the batch mean and variance
            mean = x.sum((0, )) / batch
            x_mean = x - mean.broadcast_to(x.shape)
            var = (x_mean ** 2).sum((0, )) / batch

            # Update running mean and variance (detaching to avoid tracking)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
            x_norm = x_mean / ((var + self.eps) ** 0.5).broadcast_to(x.shape)
        else:
            # Use running mean and variance during evaluation
            x_mean = x - self.running_mean.broadcast_to(x.shape)
            x_norm = x_mean / ((self.running_var.broadcast_to(x.shape) + self.eps) ** 0.5)

        return x_norm * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device = device))
        self.bias = Parameter(init.zeros(dim,device = device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        feature_number = x.shape[1]
        mean = ops.divide_scalar(ops.summation(x,axes = 1),feature_number)

        actual_mean = ops.broadcast_to(ops.reshape(mean, (x.shape[0], 1)),x.shape)

        var = ops.summation(ops.divide_scalar(ops.power_scalar(x-actual_mean,2), feature_number), axes=1)
        
        actual_var = ops.broadcast_to(ops.reshape(var, (x.shape[0], 1)), x.shape)
        
        n = ops.power_scalar(actual_var + self.eps, 0.5)

        weight = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        
        y = weight * (x - actual_mean) / n + bias
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Generate a random mask where each element has a probability of `1 - p` of being kept
            mask = (init.randb(*x.shape, p=1 - self.p,dtype="float32",device = x.device) / (1 - self.p))
            return x * mask  # Apply the mask
        else:
            # During evaluation, return the input unchanged
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
