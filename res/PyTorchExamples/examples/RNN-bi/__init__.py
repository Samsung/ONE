import torch
import torch.nn as nn

_input_size = 3
_seq_len = 2
_batch = 2
_hidden_size = 5
_num_layers = 2


# model
class net_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.RNN(_input_size, _hidden_size, _num_layers, bidirectional=True)

    def forward(self, inputs):
        return self.op(inputs[0], inputs[1])


_model_ = net_RNN()

# dummy input for onnx generation
_dummy_ = [
    torch.randn(_seq_len, _batch, _input_size),
    torch.randn(2 * _num_layers, _batch, _hidden_size)
]
