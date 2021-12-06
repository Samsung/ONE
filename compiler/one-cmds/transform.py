#!/usr/bin/python3
import onnx
import onnx.numpy_helper
import sys
import numpy as np

class ModelTransformerHelper:
  def __init__(self, model):
    self._model = model
    self._nodes_to_delete = []
    self._insert_id = 0

  def make_node(self, *p_args, **k_args):
    node = onnx.helper.make_node(*p_args, **k_args)
    self._model.graph.node.insert(self._insert_id, node)
    self._insert_id += 1
    return p_args[2]

  def make_tensor(self, tensor_data, name):
    tensor = onnx.numpy_helper.from_array(tensor_data)
    tensor.name = name
    self._model.graph.initializer.append(tensor)
    return [name]

  def mark_for_deletion(self, node):
    self._nodes_to_delete += [node]

  def get_insert_id(self):
    return self._insert_id

  def set_insert_id(self, insert_id):
    self._insert_id = insert_id

  def delete_marked_nodes(self):
    for node in self._nodes_to_delete:
      self._model.graph.node.remove(node)


class Info:
  def __init__(self, dtype, shape):
    self.dtype = dtype
    self.shape = shape


def get_tensor_infos(model):
  inferred_shape_model = onnx.shape_inference.infer_shapes(model)

  infos = {}
  for tensor in list(inferred_shape_model.graph.value_info) + list(inferred_shape_model.graph.input):
    info = Info(tensor.type.tensor_type.elem_type, [])
    for dim in tensor.type.tensor_type.shape.dim:
      info.shape += [dim.dim_value]
    infos[tensor.name] = info
    
  for tensor in list(model.graph.initializer):
    infos[tensor.name] = Info(tensor.data_type, tensor.dims)
  return infos

def dtype_to_np(dtype):
  if dtype == 1:
    return np.float32
  else:
    raise NotImplementedError('unsupported data type')

def generate_one_direction_RNN(transformer, name, X, W, R, B, initial_h, clip, activation):
  # one direction RNN:
  #
  # H = f(X*(W^T) + h*(R^T) + B)
  #
  # H  - new hidden state
  # h  - previous hidden state
  # X  - current input
  # W  - input weights matrix
  # R  - reccurent weights matrix
  # Wb - input weights matmul bias
  # Rb - reccurent weights matmul bias
  # f  - activation function

  seq_length = len(X)
  first_iter = 0
  state_tensors = []
  if initial_h is not None:
    previous_state_tensor = initial_h
  else:
    first_iter = 1
    state_tensor = transformer.make_node('Gemm', [X[0]] + W + B, [name + '_X_W_B' + str(i)], transB=1)
    if clip != None:
      state_tensor = transformer.make_node('Clip', state_tensor, [name + "_clipped"], min=-clip, max=clip)
    previous_state_tensor = transformer.make_node(activation, state_tensor, [name + '_h_state0'])
    state_tensors += previous_state_tensor

  for i in range(first_iter, seq_length):
    state_tensor = transformer.make_node('Gemm', [X[i]] + W + B, [name + '_X_W_B' + str(i)], transB=1)
    state_tensor = transformer.make_node('Gemm', previous_state_tensor + R + state_tensor, [name + '_X_W_B_plus_h_R' + str(i)], transB=1)
    if clip != None:
      state_tensor = transformer.make_node('Clip', state_tensor, [name + "_clipped"], min=-clip, max=clip)
    previous_state_tensor = transformer.make_node(activation, state_tensor, [name + '_h_state' + str(i)])
    state_tensors += previous_state_tensor
  return state_tensors


def legalize_RNN(transformer, tensor_infos, node):
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
      activations = list(map(lambda item: item.decode('UTF-8'), list(attr.strings)))
    if attr.name == 'clip':
      clip = attr.f
    if attr.name == 'direction':
      direction = attr.s.decode('UTF-8')
    if attr.name == 'hidden_size':
      hidden_size = attr.i
    if attr.name == 'layout':
      layout = attr.i

  seq_length_dim = layout
  seq_length = tensor_infos[inputs[0]].shape[seq_length_dim]
  if hidden_size == 0:
    hidden_size = tensor_infos[inputs[2]].shape[2]

  input_split_tensor = transformer.make_node('Split', [inputs[0]], [name + '_input_split' + str(i) for i in range(seq_length)], axis=seq_length_dim, split = [1] * seq_length)
  x = []
  for i in range(len(input_split_tensor)):
    input_frame_tensor = input_split_tensor[i]
    squeezed_frame_tensor = transformer.make_node('Squeeze', [input_frame_tensor], [name + '_squeezed_frame' + str(i)], axes=[0])
    x += squeezed_frame_tensor
  if direction in ['forward', 'reverse']:
    if direction == 'reverse':
      x.reverse()
    w = transformer.make_node('Squeeze', [inputs[1]], [name + '_weight'], axes=[0])
    r = transformer.make_node('Squeeze', [inputs[2]], [name + '_rec_weight'], axes=[0])
    if len(inputs) > 3 and inputs[3] != '':
      raw_bias_tensor = transformer.make_node('Squeeze', [inputs[3]], [name + "_raw_bias"], axes=[0])
      splitted_bias_tensors = transformer.make_node('Split', raw_bias_tensor, [name + '_bias_W', name + '_bias_R'], axis = 0, split=[hidden_size] * 2)
      b = transformer.make_node('Add', splitted_bias_tensors, [name + '_composed_bias'])
    else:
      data_type = dtype_to_np(tensor_infos[inputs[2]].dtype)
      b = transformer.make_tensor(np.zeros(hidden_size, dtype=data_type), name + "_zero_bias")
    if len(inputs) > 5 and inputs[5] != '':
      direction_dim = layout
      initial_h = transformer.make_node('Squeeze', [inputs[5]], [name + "_initial_state"], axes=[direction_dim])
    else:
      initial_h = None
    activation = activations[0]
    state_tensors = generate_one_direction_RNN(transformer, name, x, w, r, b, initial_h, clip, activation)
    y_direction_dim = layout + 1
    y_h_direction_dim = layout
    state_layout_tensors = []
    for state in state_tensors:
      state_layout_tensors += transformer.make_node("Unsqueeze", [state], [state + '_layout'], axes=[seq_length_dim, y_direction_dim])

    Y_h = transformer.make_node('Unsqueeze', [state_tensors[-1]], [outputs[1]], axes=[y_h_direction_dim])
    Y = transformer.make_node('Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)
  elif direction == 'bidirectional':
    w_bi = transformer.make_node('Split', [inputs[1]], [name + '_weight_f', name + '_weight_b'], axis=0, split=[1,1])
    r_bi = transformer.make_node('Split', [inputs[2]], [name + '_rec_weight_f', name + '_rec_weight_b'], axis=0, split=[1,1])
    w = []
    r = []
    for d in range(2):
      w += transformer.make_node('Squeeze', [w_bi[d]], [name + '_weight_squeezed' + str(d)], axes=[0])
      r += transformer.make_node('Squeeze', [r_bi[d]], [name + '_rec_weight_squeezed' + str(d)], axes=[0])

    b = []
    if len(inputs) > 3 and inputs[3] != '':
      raw_bias_tensors = transformer.make_node('Split', [inputs[3]], [name + "_raw_bias_f", name + "_raw_bias_b"], axis=0, split=[1, 1])
      for d in range(2):
        raw_bias_tensors_squeezed = transformer.make_node('Squeeze', [raw_bias_tensors[d]], [name + '_bias_squeezed' + str(d)], axes=[0])
        splitted_bias_tensors = transformer.make_node('Split', raw_bias_tensors_squeezed, [name + '_bias_W' + str(d), name + '_bias_R' + str(d)], axis = 0, split=[hidden_size] * 2)
        b += transformer.make_node('Add', splitted_bias_tensors, [name + '_composed_bias' + str(d)])
    else:
      data_type = dtype_to_np(tensor_infos[inputs[2]].dtype)
      b = transformer.make_tensor(np.zeros(hidden_size, dtype=data_type), name + "_zero_bias") * 2
    initial_h = [None, None]
    if len(inputs) > 5 and inputs[5] != '':
      direction_dim = layout
      initial_h = transformer.make_node('Split', [inputs[5]], [name + "_initial_state_f", name + "_initial_state_b"], axis=direction_dim, split=[1, 1])
      for d in range(2):
        initial_h[d] = transformer.make_node('Squeeze', [initial_h[d]], [name + "_initial_state_squeezed" + str(d)], axes=[direction_dim])

    state_tensors_f = generate_one_direction_RNN(transformer, name + '_f', x, [w[0]], [r[0]], [b[0]], initial_h[0], clip, activations[0])
    x.reverse()
    state_tensors_b = generate_one_direction_RNN(transformer, name + '_b', x, [w[1]], [r[1]], [b[1]], initial_h[1], clip, activations[1])

    y_direction_dim = layout + 1
    y_h_direction_dim = layout
    state_layout_tensors = []
    for t in range(seq_length):
      state_f = state_tensors_f[t]
      state_b = state_tensors_b[t]
      state_layout_tensors_f = transformer.make_node("Unsqueeze", [state_f], [state_f + '_layout'], axes=[seq_length_dim, y_direction_dim])
      state_layout_tensors_b = transformer.make_node("Unsqueeze", [state_b], [state_b + '_layout'], axes=[seq_length_dim, y_direction_dim])
      state_layout_tensors += transformer.make_node("Concat", state_layout_tensors_f + state_layout_tensors_b, [name + "_layout_state_cat" + str(t)], axis=y_direction_dim)

    Y_h = transformer.make_node('Squeeze', [state_layout_tensors[-1]], [outputs[1]], axes=[seq_length_dim])
    Y = transformer.make_node('Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)
  else:
    raise RuntimeError('Unknown RNN type')

  transformer.mark_for_deletion(node)


def legalize_model(model):
  tensor_infos = get_tensor_infos(model)

  transformer = ModelTransformerHelper(model)

  node_id = 0
  while node_id < len(model.graph.node):
    node = model.graph.node[node_id]
    if node.op_type == 'RNN':
      # opset version is required by split operation
      assert(model.opset_import[0].version < 13)
      transformer.set_insert_id(node_id)
      legalize_RNN(transformer, tensor_infos, node)
      node_id = transformer.get_insert_id()
    node_id += 1

  transformer.delete_marked_nodes()


if len(sys.argv) < 2:
  model=onnx.load("/home/binarman/rnn_experiments/RNN.onnx")
else:
  model=onnx.load(sys.argv[1])
legalize_model(model)
onnx.save(model, "/home/binarman/rnn_experiments/modified.onnx")
