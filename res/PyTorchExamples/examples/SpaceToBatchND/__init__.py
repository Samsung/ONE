import torch
import torch.nn as nn
import numpy as np


# model equivalent to tensorflow space_to_batch, but with channels first layout
class net_SpaceToBatchND(nn.Module):
    def __init__(self, block_shape, pad):
        super().__init__()
        self.block_shape = block_shape
        self.pad = pad

    def forward(self, input):
        # Prepare attributes
        input_shape = list(map(int, list(input.shape)))
        block_shape = self.block_shape
        pad = self.pad

        # number of spatial dimensions
        m = len(block_shape)
        # rest of dimensions
        n = len(input.shape) - m
        print(pad)
        # output batch size
        batch_size = input_shape[0]

        out_spatial_dim = [
            (input_shape[i + n] + pad[i * 2] + pad[i * 2 + 1]) // block_shape[i]
            for i in range(m)
        ]
        unfolded_shape = [batch_size] + input_shape[1:n] + [
            dim for i in range(m) for dim in [out_spatial_dim[i], block_shape[i]]
        ]
        fold_shape = [batch_size * np.prod(block_shape)
                      ] + input_shape[1:n] + out_spatial_dim
        permute_dims = list(range(n + 1, n + 2 * m, 2)) + list(range(n)) + list(
            range(n, n + 2 * m, 2))

        # Actual model starts here
        padded_input = torch.nn.functional.pad(input, pad)
        print(input.shape)
        print(padded_input.shape)
        unfolded_input = padded_input.reshape(unfolded_shape)
        permuted = torch.permute(unfolded_input, permute_dims)
        output = permuted.reshape(fold_shape)
        return output


_model_ = net_SpaceToBatchND([2, 2], [1, 0, 0, 1])

# dummy input for onnx generation
_dummy_ = torch.randn(2, 4, 5, 5)
