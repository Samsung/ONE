import torch
import onnx


# Generate ReduceSum operator with Float32, Rank-2
class net_ReduceSum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(input)  # This will be replaced with ReduceSumSquare

    def onnx_opset_version(self):
        return 13

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        # Find the ReduceSum node
        target = None
        for n in onnx_model.graph.node:
            if n.op_type == "ReduceSum":
                target = n
                break
        if target is None:
            raise RuntimeError("ReduceSum node not found")

        target.op_type = "ReduceSumSquare"
        target.name = "ReduceSumSquare_0"

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, model_path)


_model_ = net_ReduceSum()

_inputs_ = torch.randn(10, 12)
