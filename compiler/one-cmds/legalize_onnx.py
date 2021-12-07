#!/usr/bin/python3
import onnx
import onnx.numpy_helper
import sys
import numpy as np
import re

# Transform onnx model to make it compilable with our toolchain
# List of transformations:
# - Replace RNN operation with unrolled subgraph
# - Replace LSTM operation with unrolled subgraph

def reverse_str(s):
  return ''.join(reversed(s))

def parse_tensor_name(name):
  rev = reverse_str(name)
  m = re.match('(\d*)(.*)', rev)
  if m.groups()[0] != '':
    return (reverse_str(m.groups()[1]), int(reverse_str(m.groups()[0])))
  else:
    return (reverse_str(m.groups()[1]), 0)

class ModelTransformerHelper:
  def __init__(self, model):
    self._model = model
    self._nodes_to_delete = []
    self._insert_id = 0
    # each tensor has name containing base name and unique number. for example:
    # "abc_123": "abs_" - base name, "123" - unique number
    # if no number in name, consider it is equal to "0"

    # mapping from base names to largest given number
    self._base_name_idx = {}
    # gather name information for existing tensors
    for node in model.graph.node:
      for t in list(node.input) + list(node.output):
        base_name, number = parse_tensor_name(t)
        if base_name in self._base_name_idx:
          self._base_name_idx[base_name] = max(self._base_name_idx[base_name], number)
        else:
          self._base_name_idx[base_name] = number

  def make_tensor_with_base_name(self, base_name):
    if base_name in self._base_name_idx:
      self._base_name_idx[base_name] += 1
      return base_name + str(self._base_name_idx[base_name])
    else:
      self._base_name_idx[base_name] = 0
      return base_name + '0'

  def make_node(self, opcode, inputs, outputs, *p_args, **k_args):
    if type(outputs) == int:
      outputs = [self.make_tensor_with_base_name('') for i in range(outputs)]
    assert(type(outputs) == list)
    node = onnx.helper.make_node(opcode, inputs, outputs, *p_args, **k_args)
    self._model.graph.node.insert(self._insert_id, node)
    self._insert_id += 1
    return outputs

  def make_constant_tensor(self, tensor_data, base_name):
    tensor = onnx.numpy_helper.from_array(tensor_data)
    tensor.name = self.make_tensor_with_base_name(base_name)
    self._model.graph.initializer.append(tensor)
    return [tensor.name]

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

def generate_one_direction_RNN(transformer, X, W, R, B, initial_h, clip, activation):
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
    state_tensor = transformer.make_node('Gemm', [X[0]] + W + B, 1, transB=1)
    if clip != None:
      state_tensor = transformer.make_node('Clip', state_tensor, 1, min=-clip, max=clip)
    previous_state_tensor = transformer.make_node(activation, state_tensor, 1)
    state_tensors += previous_state_tensor

  for i in range(first_iter, seq_length):
    state_tensor = transformer.make_node('Gemm', [X[i]] + W + B, 1, transB=1)
    state_tensor = transformer.make_node('Gemm', previous_state_tensor + R + state_tensor, 1, transB=1)
    if clip != None:
      state_tensor = transformer.make_node('Clip', state_tensor, 1, min=-clip, max=clip)
    previous_state_tensor = transformer.make_node(activation, state_tensor, 1)
    state_tensors += previous_state_tensor
  return state_tensors


def transform_unidirectional_RNN(transformer, original_node, x, tensor_infos, activations, clip, direction, hidden_size, layout):
  inputs = original_node.input
  outputs = original_node.output
  if direction == 'reverse':
    x.reverse()
  w = transformer.make_node('Squeeze', [inputs[1]], 1, axes=[0])
  r = transformer.make_node('Squeeze', [inputs[2]], 1, axes=[0])
  if len(inputs) > 3 and inputs[3] != '':
    raw_bias_tensor = transformer.make_node('Squeeze', [inputs[3]], 1, axes=[0])
    splitted_bias_tensors = transformer.make_node('Split', raw_bias_tensor, 2, axis = 0, split=[hidden_size] * 2)
    b = transformer.make_node('Add', splitted_bias_tensors, 1)
  else:
    data_type = dtype_to_np(tensor_infos[inputs[2]].dtype)
    b = transformer.make_constant_tensor(np.zeros(hidden_size, dtype=data_type), "zero_bias")
  if len(inputs) > 5 and inputs[5] != '':
    direction_dim = layout
    initial_h = transformer.make_node('Squeeze', [inputs[5]], 1, axes=[direction_dim])
  else:
    initial_h = None
  activation = activations[0]
  state_tensors = generate_one_direction_RNN(transformer, x, w, r, b, initial_h, clip, activation)
  y_direction_dim = layout + 1
  y_h_direction_dim = layout
  state_layout_tensors = []
  seq_length_dim = layout
  for state in state_tensors:
    state_layout_tensors += transformer.make_node("Unsqueeze", [state], 1, axes=[seq_length_dim, y_direction_dim])

  Y_h = transformer.make_node('Unsqueeze', [state_tensors[-1]], [outputs[1]], axes=[y_h_direction_dim])
  Y = transformer.make_node('Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)


def transform_bidirectional_RNN(transformer, original_node, x, tensor_infos, activations, clip, hidden_size, layout):
  inputs = original_node.input
  outputs = original_node.output
  w_bi = transformer.make_node('Split', [inputs[1]], 2, axis=0, split=[1,1])
  r_bi = transformer.make_node('Split', [inputs[2]], 2, axis=0, split=[1,1])
  w = []
  r = []
  for d in range(2):
    w += transformer.make_node('Squeeze', [w_bi[d]], 1, axes=[0])
    r += transformer.make_node('Squeeze', [r_bi[d]], 1, axes=[0])

  b = []
  if len(inputs) > 3 and inputs[3] != '':
    raw_bias_tensors = transformer.make_node('Split', [inputs[3]], 2, axis=0, split=[1, 1])
    for d in range(2):
      raw_bias_tensors_squeezed = transformer.make_node('Squeeze', [raw_bias_tensors[d]], 1, axes=[0])
      splitted_bias_tensors = transformer.make_node('Split', raw_bias_tensors_squeezed, 2, axis = 0, split=[hidden_size] * 2)
      b += transformer.make_node('Add', splitted_bias_tensors, 1)
  else:
    data_type = dtype_to_np(tensor_infos[inputs[2]].dtype)
    b = transformer.make_constant_tensor(np.zeros(hidden_size, dtype=data_type), "zero_bias") * 2
  initial_h = [None, None]
  if len(inputs) > 5 and inputs[5] != '':
    direction_dim = layout
    initial_h = transformer.make_node('Split', [inputs[5]], 2, axis=direction_dim, split=[1, 1])
    for d in range(2):
      initial_h[d] = transformer.make_node('Squeeze', [initial_h[d]], 1, axes=[direction_dim])

  state_tensors_f = generate_one_direction_RNN(transformer, x, [w[0]], [r[0]], [b[0]], initial_h[0], clip, activations[0])
  x.reverse()
  state_tensors_b = generate_one_direction_RNN(transformer, x, [w[1]], [r[1]], [b[1]], initial_h[1], clip, activations[1])

  y_direction_dim = layout + 1
  y_h_direction_dim = layout
  state_layout_tensors = []
  seq_length_dim = layout
  seq_length = len(x)
  for t in range(seq_length):
    state_f = state_tensors_f[t]
    state_b = state_tensors_b[t]
    state_layout_tensors_f = transformer.make_node("Unsqueeze", [state_f], 1, axes=[seq_length_dim, y_direction_dim])
    state_layout_tensors_b = transformer.make_node("Unsqueeze", [state_b], 1, axes=[seq_length_dim, y_direction_dim])
    state_layout_tensors += transformer.make_node("Concat", state_layout_tensors_f + state_layout_tensors_b, 1, axis=y_direction_dim)

  Y_h = transformer.make_node('Squeeze', [state_layout_tensors[-1]], [outputs[1]], axes=[seq_length_dim])
  Y = transformer.make_node('Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)

def legalize_RNN(transformer, tensor_infos, node):
  inputs = node.input
  outputs = node.output
  if len(inputs) > 4 and inputs[4] != '':
    raise NotImplementedError('Variadic length of output is not supported')
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

  for act in activations:
    if act not in ['Relu', 'Tanh', 'Sigmoid']:
      raise NotImplementedError('Unsupported activation function')

  seq_length_dim = layout
  seq_length = tensor_infos[inputs[0]].shape[seq_length_dim]
  if hidden_size == 0:
    hidden_size = tensor_infos[inputs[2]].shape[2]

  input_split_tensor = transformer.make_node('Split', [inputs[0]], seq_length, axis=seq_length_dim, split = [1] * seq_length)
  x = []
  for i in range(len(input_split_tensor)):
    input_frame_tensor = input_split_tensor[i]
    squeezed_frame_tensor = transformer.make_node('Squeeze', [input_frame_tensor], 1, axes=[0])
    x += squeezed_frame_tensor

  if direction in ['forward', 'reverse']:
    transform_unidirectional_RNN(transformer, node, x, tensor_infos, activations, clip, direction, hidden_size, layout)
  elif direction == 'bidirectional':
    transform_bidirectional_RNN(transformer, node, x, tensor_infos, activations, clip, hidden_size, layout)
  else:
    raise RuntimeError('Unknown RNN type')

  transformer.mark_for_deletion(node)


def legalize_LSTM(transformer, tensor_infos, node):
  pass


def legalize_model(model):
  tensor_infos = get_tensor_infos(model)

  transformer = ModelTransformerHelper(model)

  node_id = 0
  while node_id < len(model.graph.node):
    node = model.graph.node[node_id]
    if node.op_type == 'RNN':
      # opset version is required by split operation
      if model.opset_import[0].version >= 13:
        raise NotImplementedError('Can not generate code with opcode version 13 and greater')
      transformer.set_insert_id(node_id)
      legalize_RNN(transformer, tensor_infos, node)
      node_id = transformer.get_insert_id()
    elif node.op_type == 'LSTM':
      if model.opset_import[0].version >= 13:
        raise NotImplementedError('Can not generate code with opcode version 13 and greater')
      transformer.set_insert_id(node_id)
      legalize_LSTM(transformer, tensor_infos, node)
      node_id = transformer.get_insert_id()
    node_id += 1

  transformer.delete_marked_nodes()


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('usage: ./legalize_onnx.py <path to input model> <path to output model>')
    exit(1)
  model=onnx.load(sys.argv[1])
  legalize_model(model)
  onnx.save(model, sys.argv[2])
