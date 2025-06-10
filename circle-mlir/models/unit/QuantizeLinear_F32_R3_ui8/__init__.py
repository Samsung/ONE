import torch

UINT8_MIN = 0
UINT8_MAX = 255


# Generate QuantizeLinear/DequantizeLinear operator with Float32, Rank-3, layer wise, UINT8
class net_QuantizeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        scale = 0.001
        zero_point = 127
        return torch.fake_quantize_per_tensor_affine(input, scale, zero_point, UINT8_MIN,
                                                     UINT8_MAX)

    def onnx_opset_version(self):
        # TODO set version
        return 13


_model_ = net_QuantizeLinear()

_inputs_ = torch.randn(1, 16, 16)
