#!/usr/bin/python3

import onnx

model=onnx.load("/home/binarman/rnn_experiments/RNN.onnx")

inferred_shape_model = onnx.shape_inference.infer_shapes(model)

shapes = {}
for tensor in list(inferred_shape_model.graph.value_info) + list(inferred_shape_model.graph.input):
  shape = []
  for dim in tensor.type.tensor_type.shape.dim:
    shape += [dim.dim_value]
  shapes[tensor.name] = shape

nodes_to_delete = []

for node_id in range(len(model.graph.node)):
  node = model.graph.node[node_id]
  if node.op_type == 'RNN':
    # one direction RNN:
    #
    # H = f(X*(W^T) + h*(R^T) + Wb + Rb)
    #
    # H  - new hidden state
    # h  - previous hidden state
    # X  - current input
    # W  - input weights matrix
    # R  - reccurent weights matrix
    # Wb - input weights matmul bias
    # Rb - reccurent weights matmul bias
    # f  - activation function
    inputs = node.input
    outputs = node.output
    name = node.name
    # attributes
    activation_alpha = []
    activation_beta = []
    activations = ['Tanh', 'Tanh']
    clip = None
    direction = 'forward'
    hidden_size = 0
    layout = 0

    for attr in node.attribute:
      if attr.name == 'activation_alpha':
        activation_alpha = attr.floats
      if attr.name == 'activation_beta':
        activation_beta = attr.floats
      if attr.name == 'activations':
        activations = attr.strings
      if attr.name == 'clip':
        clip = attr.f
      if attr.name == 'direction':
        direction = attr.s
      if attr.name == 'hidden_size':
        hidden_size = attr.i
      if attr.name == 'layout':
        layout = attr.i

    seq_length_dim = layout
    seq_length = shapes[inputs[0]][seq_length_dim]

    # opset version is required by split operation
    assert(model.opset_import[0].version < 13)
    assert(direction == 'forward')
    assert(activations[0].decode('UTF-8') == 'Tanh')

    split_inputs = [inputs[0]]
    split_outputs = [name + '_input_split' + str(i) for i in range(seq_length)]
    split_segments = [1] * seq_length

    input_split = onnx.helper.make_node('Split', split_inputs, split_outputs, axis=seq_length_dim, split = split_segments)
    model.graph.node.append(input_split)

    squeezed_weights = onnx.helper.make_node('Squeeze', [inputs[1]], [name + '_weight'], axes=[0])
    model.graph.node.append(squeezed_weights)

    squeezed_rec_weights = onnx.helper.make_node('Squeeze', [inputs[2]], [name + '_rec_weight'], axes=[0])
    model.graph.node.append(squeezed_rec_weights)

    bias_name = ''

    previous_state = inputs[5]
    if previous_state == '':
      # todo unroll first iteration?
      assert(False)

    for i in range(seq_length):
      input_squeeze = [name + '_input_split' + str(i)]
      output_squeeze = [name + '_inputs_squeeze' + str(i)]
      sqz = onnx.helper.make_node('Squeeze', input_squeeze, output_squeeze, axes=[seq_length_dim])
      model.graph.node.append(sqz)
      
      input_mult = onnx.helper.make_node('Gemm', [output_squeeze[0], name + '_weight', bias_name], [name + '_X_W' + str(i)], transB=1, broadcast=1)
      model.graph.node.append(input_mult)

      reccurent_addition = onnx.helper.make_node('Gemm', [previous_state, name + '_rec_weight', name + '_X_W' + str(i)], [name + '_X_W_plus_h_R' + str(i)], transB=1)
      model.graph.node.append(reccurent_addition)
      
      activation = onnx.helper.make_node(activations[0].decode('UTF-8'), [name + '_X_W_plus_h_R' + str(i)], [name + 'h_state' + str(i)])
      model.graph.node.append(activation)

      previous_state = name + 'h_state' + str(i)

    Y_h = onnx.helper.make_node('Unsqueeze', [previous_state], [outputs[0]], axis=0)
    model.graph.node.append(Y_h)

    Y = onnx.helper.make_node('Concat', [name + 'h_state' + str(i) for i in range(seq_length)], [outputs[1]], axis=0)
    model.graph.node.append(Y)
    nodes_to_delete += [node]

for node in nodes_to_delete:
  model.graph.node.remove(node)

onnx.save(model, "/home/binarman/rnn_experiments/modified.onnx")
