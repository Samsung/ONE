import torch


# Generate Slice operator with Float32, Rank-3 with axes from input
class net_Slice(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1, input2):
        input = input0
        starts = input1
        ends = input2
        return input[:, :, starts:ends:]

    def onnx_opset_version(self):
        return 13


_model_ = net_Slice()

vals = [1]
vale = [1]
_inputs_ = (torch.randn(2, 8, 6), torch.tensor(vals), torch.tensor(vale))

_io_names_ = [['input', 'starts', 'ends'], ['out']]
_dynamic_axes_ = {'input': {0: '?'}}
