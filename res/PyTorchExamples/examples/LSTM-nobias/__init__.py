import torch
import torch.nn as nn

_seq_length = 2
_batch_size = 5
_input_size = 15
_hidden_size = 10
_number_layers = 1

# model
class net_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.LSTM(_input_size, _hidden_size, _number_layers, bias=False)

    def forward(self, inputs):
        return self.op(inputs[0], (inputs[1], inputs[2]))


_model_ = net_LSTM()

# dummy input for onnx generation
_dummy_ = [torch.randn(_seq_length, _batch_size, _input_size), torch.randn(_number_layers, _batch_size, _hidden_size), torch.randn(_number_layers, _batch_size, _hidden_size)]
