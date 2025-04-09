import torch

_seq_length = 1
_batch_size = 1
_input_size = 1
_hidden_size = 1
_number_layers = 1


# Generate net_LSTM operator with Float32, Rank-3, no bias
# NOTE this fails to generate
class net_LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.LSTM(_input_size, _hidden_size, _number_layers, bias=False)

    def forward(self, inputs):
        return self.op(inputs[0], (inputs[1], inputs[2]))

    def onnx_opset_version(self):
        return 11


_model_ = net_LSTM()

_inputs_ = [
    torch.randn(_seq_length, _batch_size, _input_size),
    torch.randn(_number_layers, _batch_size, _hidden_size),
    torch.randn(_number_layers, _batch_size, _hidden_size)
]
