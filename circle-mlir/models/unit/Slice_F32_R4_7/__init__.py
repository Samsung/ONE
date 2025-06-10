import torch
import onnx


# Generate Slice operator with Float32, Rank-4
class net_Slice(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # The slicing operation input[:, 0:12, 0:12, :] generates two separate Slice nodes,
        # one for each axis (axes 1 and 2).
        # To consolidate this into a single Slice node, we first generate a partial slice
        # (e.g., input[:, 0:12, :, :]) and use the post_process() function to adjust its parameters.
        # This ensures that the slicing is executed as a single operation across multiple axes.
        return input[:, 0:12, :, :]

    def onnx_opset_version(self):
        return 14

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)

        for node in onnx_model.graph.node:
            if node.op_type == "Slice":
                new_axes = [1, 2]
                new_starts = [0, 0]
                new_ends = [12, 12]
                new_steps = [1, 1]

                starts_initializer = onnx.helper.make_tensor(
                    name="starts",
                    data_type=onnx.TensorProto.INT64,
                    dims=(len(new_starts), ),
                    vals=new_starts)
                ends_initializer = onnx.helper.make_tensor(
                    name="ends",
                    data_type=onnx.TensorProto.INT64,
                    dims=(len(new_ends), ),
                    vals=new_ends)
                axes_initializer = onnx.helper.make_tensor(
                    name="axes",
                    data_type=onnx.TensorProto.INT64,
                    dims=(len(new_axes), ),
                    vals=new_axes)
                steps_initializer = onnx.helper.make_tensor(
                    name="steps",
                    data_type=onnx.TensorProto.INT64,
                    dims=(len(new_steps), ),
                    vals=new_steps)

                # Append the new initializers to the graph
                onnx_model.graph.initializer.extend([
                    starts_initializer, ends_initializer, axes_initializer,
                    steps_initializer
                ])

                # Modify the Slice node to use the new parameters
                node.input[1] = "starts"
                node.input[2] = "ends"
                node.input[3] = "axes"
                node.input[4] = "steps"

        # Remove old parameters of Slice
        contRemove = True
        didRemove = False
        while contRemove:
            didRemove = False
            for node in onnx_model.graph.node:
                if node.op_type == 'Constant':
                    onnx_model.graph.node.remove(node)
                    didRemove = True
                    break
            if not didRemove:
                contRemove = False

        # Modify the graph output to match the expected shape
        for output in onnx_model.graph.output:
            output.type.tensor_type.shape.dim[2].dim_value = 12

        onnx.save(onnx_model, model_path)


_model_ = net_Slice()

_inputs_ = torch.randn(2, 16, 16, 4)
