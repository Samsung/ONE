import torch
import onnx


# Generate network of Shape-Gather-Unsqueeze-Concat-Expand with Float32, Rank-3
# NOTE this sequence is from our target model
# NOTE counldn't find a way to produce this sequence with pytorch so,
#      after simple model is created, network is modified to produce this.
#      inputs[0] is I64 which is dtype for Concat and then replaced with F32
#      which is our target model input dtype.
# NOTE all the string names are first got from Netron tool and used here.
#      so if the network is changed, names should also be updated as so.
class net_SgucExpand(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = torch.tensor([1], dtype=torch.int64)
        self.C3 = torch.tensor([4], dtype=torch.int64)
        self.index = 1

    def forward(self, inputs):
        ginput = inputs[0]
        gather = ginput[self.index]
        unsqueeze = torch.unsqueeze(gather, 0)
        return torch.cat((self.C1, unsqueeze, self.C3)), inputs[1]

    def onnx_opset_version(self):
        return 11

    def post_process(self, model_path):
        onnx_model = onnx.load(model_path)
        # graph I/O
        #   input: [ 'onnx::Gather_0', 'onnx::Identity_1' ]
        #   output: [ '7', '8' ]
        #
        # modify graph to
        # [input]-Shape-Gather-Unsqueeze-Concat-Expand-[output]

        print(onnx_model.graph)

        # insert shape with float input
        node_shape = onnx.helper.make_node('Shape', ['onnx::Identity_1'], ['out_Shape'],
                                           name='/Shape')
        onnx_model.graph.node.insert(0, node_shape)

        # set Gather input to Shape node
        # rename Concat name output
        for node in onnx_model.graph.node:
            if node.op_type == 'Gather':
                node.input[0] = 'out_Shape'
            if node.op_type == 'Concat':
                node.output[0] = 'out_Concat'

        # Constant as 'input' of Expand node
        tensor_ein = onnx.helper.make_tensor(name='/ExpandInput',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=[1, 1, 4],
                                             vals=[1, 2, 3, 4])
        node_ein = onnx.helper.make_node('Constant', [], ['ExpandInput'],
                                         value=tensor_ein)
        onnx_model.graph.node.insert(100, node_ein)

        node_expand = onnx.helper.make_node('Expand', ['ExpandInput', 'out_Concat'],
                                            ['out_Expand'],
                                            name='/Expand')
        onnx_model.graph.node.insert(101, node_expand)

        # remove dummy identity
        for node in onnx_model.graph.node:
            if node.name == 'Identity_6':
                onnx_model.graph.node.remove(node)
                break

        # remove unused input
        for input in onnx_model.graph.input:
            if input.name == 'onnx::Gather_0':
                onnx_model.graph.input.remove(input)
                break

        # update output information
        info = onnx.helper.make_tensor_value_info('out_Expand', onnx.TensorProto.FLOAT,
                                                  [1, 1, 4])
        onnx_model.graph.output.insert(0, info)

        remove_outputs = []
        for output in onnx_model.graph.output:
            if output.name == 'out_Expand':
                continue
            remove_outputs.append(output)

        for output in remove_outputs:
            onnx_model.graph.output.remove(output)

        onnx.save(onnx_model, model_path)


_model_ = net_SgucExpand()

_inputs_ = [torch.tensor([1, 1, 4], dtype=torch.int64), torch.randn(1, 1, 4)]
