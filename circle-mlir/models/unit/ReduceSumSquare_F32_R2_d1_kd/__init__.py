import torch
import onnx


# Generate ReduceSumSquare operator with Float32, Rank-2, dim=1, keepdim=True
class net_ReduceSumSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sum(
            input)  # This will be replaced with ReduceSumSquare(axes=1, keepdim=True)

    def onnx_opset_version(self):
        return 13

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        # Find the ReduceSum node and its index
        idx = None
        target = None
        for i, n in enumerate(onnx_model.graph.node):
            if n.op_type == "ReduceSum":
                idx = i
                target = n
                break
        if target is None:
            raise RuntimeError("ReduceSum node not found")

        data_in = target.input[0]
        data_out = target.output[0]

        # Replace node with ReduceSumSquareV13
        new_node = onnx.helper.make_node(
            "ReduceSumSquare",
            inputs=[data_in],
            outputs=[data_out],
            name="ReduceSumSquare_0",
            axes=[1],  # dim=1
            keepdims=1,  # keepdim=True
        )
        onnx_model.graph.node[idx].CopyFrom(new_node)

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, model_path)


_model_ = net_ReduceSumSquare()

_inputs_ = torch.randn(10, 12)
