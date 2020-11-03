import torch
import torch.nn as nn


# model
class net_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.RNN(2, 2, 1)

    def forward(self, inputs):
        return self.op(inputs[0], inputs[1])


_model_ = net_RNN()

# dummy input for onnx generation
_dummy_ = [torch.randn(2, 2, 2), torch.randn(1, 2, 2)]

# Note: this model has problem when converting ONNX to TensorFlow
