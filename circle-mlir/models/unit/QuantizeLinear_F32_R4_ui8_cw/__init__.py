import torch

UINT8_MIN = 0
UINT8_MAX = 255

CHN_NUM = 4
CHN_AXIS = 1


# Generate QuantizeLinear/DequantizeLinear operator with Float32, Rank-4, channel-wise, UINT8
class net_QuantizeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = (torch.randn(CHN_NUM) + 1) * 0.05
        self.zero_points = (torch.randn(CHN_NUM) * 5 + 127).to(torch.int32)

    def forward(self, input):
        return torch.fake_quantize_per_channel_affine(input, self.scales,
                                                      self.zero_points, CHN_AXIS,
                                                      UINT8_MIN, UINT8_MAX)

    def onnx_opset_version(self):
        # TODO set version
        return 13


_model_ = net_QuantizeLinear()

_inputs_ = torch.randn(1, CHN_NUM, 16, 9)
