import torch
import torch.nn as nn
import numpy as np


# model equivalent to tensorflow batch_to_space, but with channels first layout
class net_BatchToSpaceND(nn.Module):
    def __init__(self, block_shape, crop):
        super().__init__()
        self.block_shape = block_shape
        self.crop = crop

    def forward(self, input):
        # Prepare attributes
        input_shape = list(map(int, list(input.shape)))
        block_shape = self.block_shape
        crop = self.crop

        # number of spatial dimensions
        m = len(block_shape)
        # rest of dimensions
        n = len(input.shape) - m
        # output batch size
        batch_size = input_shape[0] // np.product(block_shape)

        unfolded_shape = list(block_shape) + [batch_size] + input_shape[1:]
        fold_shape = [batch_size] + input_shape[1:n] + [
            input_shape[i + n] * block_shape[i] for i in range(m)
        ]
        permute_dims = list(range(
            m, m + n)) + [i + mod for i in range(m) for mod in [n + m, 0]]

        # Actual model starts here
        unfolded_input = input.reshape(unfolded_shape)
        permuted = torch.permute(unfolded_input, permute_dims)
        full_output = permuted.reshape(fold_shape)
        # crop output tensor
        crop_output = full_output
        for i in range(m):
            crop_size = sum(crop[i])
            crop_output = crop_output.narrow(i + n, crop[i][0],
                                             fold_shape[i + n] - crop_size)
        return crop_output


_model_ = net_BatchToSpaceND([2, 2], [[1, 0], [0, 1]])

# dummy input for onnx generation
_dummy_ = torch.randn(8, 4, 3, 3)
