import torch
import onnx


# Generate Expand operator with Float32, Rank-2, unknown
class net_Expand(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs[0].expand(inputs[1].shape)

    def onnx_opset_version(self):
        # TODO set to appropriate value
        return 9


_model_ = net_Expand()

_inputs_ = [torch.Tensor(1, 4), torch.Tensor(3, 4)]

# refer https://github.com/onnx/onnx/issues/654#issuecomment-521233285
_io_names_ = [['input'], ['output']]
_dynamic_axes_ = {'output': {0: '?'}}
