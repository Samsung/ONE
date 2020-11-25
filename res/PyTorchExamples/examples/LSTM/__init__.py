import torch
import torch.nn as nn


# model
class net_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.LSTM(10, 20, 1)

    def forward(self, inputs):
        return self.op(inputs[0], (inputs[1], inputs[2]))


_model_ = net_LSTM()

# dummy input for onnx generation
_dummy_ = [torch.randn(5, 3, 10), torch.randn(1, 3, 20), torch.randn(1, 3, 20)]

# Note: this model has problem when converting ONNX to TensorFlow
