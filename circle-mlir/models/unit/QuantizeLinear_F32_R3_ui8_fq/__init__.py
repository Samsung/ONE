import torch


# Generate QuantizeLinear/DequantizeLinear operator with Float32, Rank-3, layer wise, UINT8
# with torch.quantization.FakeQuantize() API
class net_QuantizeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_quant = torch.quantization.FakeQuantize()
        self.fake_quant.disable_observer()

    def forward(self, input):
        return self.fake_quant(input)

    def onnx_opset_version(self):
        # TODO set version
        return 13


_model_ = net_QuantizeLinear()

_inputs_ = torch.randn(1, 1, 16)

# NOTE input of (1, 16, 16) gives error
#      "Cannot reshape a tensor with 1 elements to shape [1,16,1] (16 elements)"
