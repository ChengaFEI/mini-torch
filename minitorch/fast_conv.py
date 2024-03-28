from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # Step 1: locate each cell in the output tensor
    for batch_i in prange(batch):
        for out_channel_i in prange(out_channels):
            for out_width_i in prange(out_width):
                # Step 2: init the accumulated value
                accu = 0.

                # Step 3: iteratively sum up the product of input and weight
                for in_channel_i in prange(in_channels):
                    for weight_i in prange(kw):
                        # Step 3.1: get the position of the input cell
                        if reverse:
                            in_width_i = out_width_i - weight_i
                            input_pos = batch_i * input_strides[0] + \
                                        in_channel_i * input_strides[1] + \
                                        in_width_i * input_strides[2]
                        else:
                            in_width_i = out_width_i + weight_i
                            input_pos = batch_i * input_strides[0] + \
                                        in_channel_i * input_strides[1] + \
                                        in_width_i * input_strides[2]

                        # Step 3.2: get the position of the kernel cell
                        weight_pos = out_channel_i * weight_strides[0] + \
                                     in_channel_i * weight_strides[1] + \
                                     weight_i * weight_strides[2]

                        # Step 3.3: multiply the input cell and the kernel cell
                        if 0 <= in_width_i < width:
                            accu += input[input_pos] * weight[weight_pos]

                # Step 4: get the position of the output cell
                out_pos = batch_i * out_strides[0] + \
                          out_channel_i * out_strides[1] + \
                          out_width_i * out_strides[2]

                # Step 5: assign the accumulated value to the output cell
                out[out_pos] = accu

tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """

    out_batches, out_channels, out_height, out_width = out_shape
    in_batches, in_channels, in_height, in_width = input_shape
    weight_out_channels, weight_in_channels, weight_height, weight_width = weight_shape

    assert out_batches == in_batches and \
           out_channels == weight_out_channels and \
           in_channels == weight_in_channels
    
    # Step 1: locate each cell in the output tensor
    for out_batch_i in prange(out_batches):
        in_batch_i = out_batch_i
        for out_channel_i in prange(out_channels):
            for out_height_i in prange(out_height):
                for out_width_i in prange(out_width):
                    # Step 2: init the accumulated value
                    accu = 0.

                    # Step 3: iteratively sum up the product of the input and the weight
                    for in_channel_i in prange(in_channels):
                        for weight_height_i in prange(weight_height):
                            for weight_width_i in prange(weight_width):
                                # Step 3.1: find the position of the input cell
                                if reverse:
                                    in_height_i = out_height_i - weight_height_i
                                    in_width_i = out_width_i - weight_width_i
                                else:
                                    in_height_i = out_height_i + weight_height_i
                                    in_width_i = out_width_i + weight_width_i

                                in_pos = in_batch_i * input_strides[0] + \
                                         in_channel_i * input_strides[1] + \
                                         in_height_i * input_strides[2] + \
                                         in_width_i * input_strides[3]

                                # Step 3.2: find the position of the weight cell
                                weight_pos = out_channel_i * weight_strides[0] + \
                                             in_channel_i * weight_strides[1] + \
                                             weight_height_i * weight_strides[2] + \
                                             weight_width_i * weight_strides[3]

                                # Step 3.3: multiply the input cell and the weight cell if in bound
                                if 0 <= in_height_i < in_height and 0 <= in_width_i < in_width:
                                    accu += input[in_pos] * weight[weight_pos]

                    # Step 4: find the position of the output cell
                    out_pos = out_batch_i * out_strides[0] + \
                              out_channel_i * out_strides[1] + \
                              out_height_i * out_strides[2] + \
                              out_width_i * out_strides[3]

                    # Step 5: assign the accumulated value in the output cell
                    out[out_pos] = accu

tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
