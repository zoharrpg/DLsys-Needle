"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = b * (a ** (b - 1)) * out_grad
        grad_b = (a ** b * log(a)) * out_grad
        return grad_a, grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Calculate the gradient using NDArray's power and multiplication operations
        return out_grad * self.scalar * power_scalar(node.inputs[0],self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs

        grad_a = out_grad / b

        grad_b = -out_grad * a / power_scalar(b,2)
        return grad_a,grad_b
        
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # Default to swapping the last two dimensions
            permuted_axes = list(range(a.ndim))
            permuted_axes[-2], permuted_axes[-1] = permuted_axes[-1], permuted_axes[-2]
        else:
            # Permute based on specified axes
            permuted_axes = list(range(a.ndim))
            axis1, axis2 = self.axes
            permuted_axes[axis1], permuted_axes[axis2] = permuted_axes[axis2], permuted_axes[axis1]

        return a.permute(permuted_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        original_shape = node.inputs[0].shape
        return out_grad.reshape(original_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if a.shape == self.shape:
            return a
        return array_api.broadcast_to(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        input_shape = a.shape
        output_shape = out_grad.shape
        grad = out_grad
        for i in range(len(output_shape) - len(input_shape)):
          grad = summation(grad, axes=0)
        
        exp = []
        for i, dim in enumerate(input_shape):
          if dim == 1:
            exp.append(i)
        
        return reshape(summation(grad, tuple(exp)), input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum(axis = None)
        elif isinstance(self.axes, int) or (isinstance(self.axes, (list, tuple)) and len(self.axes) == 1):
            return a.sum(self.axes)
        else:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # input_shape = node.inputs[0].shape

        # # Handle case when summing over axes
        # if self.axes is not None:
        #     axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        #     # Reshape out_grad to expand back over the summed axes
        #     out_grad = reshape(out_grad,tuple(1 if i in axes else s for i, s in enumerate(input_shape)))

        # # Broadcast the gradient to the input shape
        # return broadcast_to(out_grad, input_shape)
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Type not support!")
        
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs

        # Compute gradient w.r.t 'a'
        grad_a = matmul(out_grad, transpose(b))
        # Adjust grad_a to match the shape of 'a'
        while len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes=0)
        for i in range(len(a.shape)):
            if a.shape[i] == 1 and grad_a.shape[i] != 1:
                grad_a = summation(grad_a, axes=i, keepdims=True)


        # Compute gradient w.r.t 'b'
        grad_b = matmul(transpose(a), out_grad)
      # Adjust grad_b to match the shape of 'b'
        while len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes=0)
        for i in range(len(b.shape)):
            if b.shape[i] == 1 and grad_b.shape[i] != 1:
              grad_b = summation(grad_b, axes=i, keepdims=True)

        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]

        relu_grad = a.realize_cached_data() > 0

        return out_grad * relu_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        tanh_a = node.realize_cached_data()  # Get the result of tanh(a)
        grad = 1 - tanh_a ** 2
        return out_grad * grad  # Chain rule
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION

        shape = args[0].shape

        new_shape = list(shape)
        
        new_shape.insert(self.axis,len(args))

        out = array_api.empty(new_shape, device=args[0].device,dtype=args[0].dtype)

        slices = [slice(0, sha) for sha in new_shape]

        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr
        return out




        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad,self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        N = A.shape[self.axis] 
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []
        for i in range(N):
            slices[self.axis] = slice(i, i+1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        
        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)

        for axis in self.axes:
          if axis >= len(a.shape):
                continue
          new_shape[axis] = new_shape[axis] * (self.dilation + 1)

        result = array_api.full(new_shape,0,device = a.device)
        index_slices = [ slice(0, new_shape[axe], self.dilation+1) if axe in self.axes else slice(0, new_shape[axe], 1)for axe in range(len(a.shape))]
        
        result[tuple(index_slices)] = a
        return result
        
        #result = np.zero(new_shape,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        index_slices = [
            slice(0, a.shape[axis], self.dilation + 1) if axis in self.axes else slice(0, a.shape[axis])
            for axis in range(len(a.shape))
        ]

        # Return the undilated view of the tensor
        return a[tuple(index_slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        
        # Calculate the output dimensions after applying stride and padding
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in

        # Create the sliding window view with strides adjusted for stride parameter
        A_strided = A.as_strided(
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        ).compact().reshape((N * H_out * W_out, inner_dim))

        # Perform matrix multiplication with reshaped kernel
        B_reshaped = B.compact().reshape((inner_dim, C_out))
        out = A_strided @ B_reshaped

        # Reshape output to expected 4D shape
        return out.reshape((N, H_out, W_out, C_out))



        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs

        # Compute the gradient with respect to the input A
        # Step 1: Dilate the output gradient
        if self.stride > 1:
          out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        B_permute = transpose(flip(B, (0, 1)), (2, 3))

        A_grad = conv(out_grad,B_permute,padding =B.shape[0]-1 - self.padding)

        A_permute = transpose(A, (0, 3))
        
        out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        B_grad_transpose = conv(A_permute, out_grad_permute, padding=self.padding)
        
        B_grad = transpose(transpose(B_grad_transpose, (0, 1)), (1, 2))
        
        return A_grad, B_grad
        
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


