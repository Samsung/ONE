import torch
import torch.nn as nn

_input_size = 4
_seq_len = 2
_batch = 3
_hidden_size = 3


# model
class net_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.RNN(_input_size, _hidden_size, 1, nonlinearity='relu')

    def forward(self, inputs):
        return self.op(inputs[0], inputs[1])


_model_ = net_RNN()

# dummy input for onnx generation
_dummy_ = [
    torch.randn(_seq_len, _batch, _input_size),
    torch.randn(1, _batch, _hidden_size)
]
