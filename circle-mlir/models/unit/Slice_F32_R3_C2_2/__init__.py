import torch
import onnx
import json


# Generate lower and upper limit of test value in the input node.
# low : lower limit of the input value (inclusive)
# high : upper limit of the input value (exclusive)
def make_input_test_limit(input, low, high):
    data = {"low": low, "high": high}
    input.doc_string = json.dumps(data)


# Generate Slice operator with Float32, Rank-3, non-const starts/ends, const axes/steps
class net_Slice(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1, input2):
        return input0, input1, input2

    def onnx_opset_version(self):
        return 12

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)
        # graph I/O
        #   input: [ 'onnx::Identity_0', 'onnx::Identity_1', 'onnx::Identity_2' ]
        #   output: [ '3', '4', '5' ]
        #
        # modify graph to
        #   [Data, Starts, Ends]-Slice-[output]

        for input in onnx_model.graph.input:
            if input.name == 'onnx::Identity_0':
                input.name = 'Data'
            if input.name == 'onnx::Identity_1':
                input.name = 'Starts'
                make_input_test_limit(input, 0, 8)
            if input.name == 'onnx::Identity_2':
                input.name = 'Ends'
                make_input_test_limit(input, -4, 0)  # negative input test

        # Constant as 'axes' of Expand node
        axes = onnx.helper.make_tensor(name='/Axes',
                                       data_type=onnx.TensorProto.INT64,
                                       dims=[1],
                                       vals=[-1])
        node_axes = onnx.helper.make_node('Constant', [], ['Axes'], value=axes)
        onnx_model.graph.node.insert(0, node_axes)

        # Constant as 'steps' of Expand node
        steps = onnx.helper.make_tensor(name='/Steps',
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[1],
                                        vals=[1])
        node_steps = onnx.helper.make_node('Constant', [], ['Steps'], value=steps)
        onnx_model.graph.node.insert(1, node_steps)

        # Create SliceOp node
        slice_node = onnx.helper.make_node(
            'Slice',
            inputs=['Data', 'Starts', 'Ends', 'Axes', 'Steps'],
            outputs=['output'])
        onnx_model.graph.node.insert(2, slice_node)

        # Update output information
        out = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT,
                                                 [2, 2, 0])
        onnx_model.graph.output.insert(102, out)

        # Remove dummy identity
        identities_to_remove = ['Identity_0', 'Identity_1', 'Identity_2']
        nodes_to_remove = []
        for node in onnx_model.graph.node:
            if node.name in identities_to_remove:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            onnx_model.graph.node.remove(node)

        remove_outputs = []
        for output in onnx_model.graph.output:
            if output.name == 'output':
                continue
            remove_outputs.append(output)

        for output in remove_outputs:
            onnx_model.graph.output.remove(output)

        onnx.save(onnx_model, model_path)


_model_ = net_Slice()

_inputs_ = (torch.randn(2, 2, 12), torch.tensor([4], dtype=torch.int64),
            torch.tensor([8], dtype=torch.int64))
