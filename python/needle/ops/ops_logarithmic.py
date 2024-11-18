from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(self.axes,keepdims=True)
        end_max_z = Z.max(self.axes)

        return array_api.log(array_api.sum(array_api.exp(Z-max_z.broadcast_to(Z.shape)),axis=self.axes)) + end_max_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]

        max_z = Tensor(z.realize_cached_data().max(axis = self.axes, keepdims=True),device = z.device)
        exp_z = exp(z - max_z.broadcast_to(z.shape))

        sum_exp_z = summation(exp_z, self.axes)

        grad = out_grad / sum_exp_z

        axes = (self.axes,) if isinstance(self.axes, int) else self.axes

        exp_shape = list(z.shape)
        
        #axes = range(len(exp_shape)) if self.axes is None else self.axes
        for axis in axes:
          exp_shape[axis] = 1
        grad = grad.reshape(exp_shape).broadcast_to(z.shape)
        grad = grad * exp_z
        return grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

