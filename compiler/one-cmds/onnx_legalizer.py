#!/usr/bin/python3

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
                    self._base_name_idx[base_name] = max(self._base_name_idx[base_name],
                                                         number)
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
        """
        Create node and insert it in graph.

        Parameters:
            opcode (str): opcode name of desired operation
            inputs (list of str): names of input tensors
            outputs (list of str or int): names of output tensors or number of tensors that should be created
            p_args: additional arguments for onnx make_node helper
            k_args: attributes for onnx node
        """
        if type(outputs) == int:
            outputs = [self.make_tensor_with_base_name('') for i in range(outputs)]
        assert (type(outputs) == list)
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
    for tensor in list(inferred_shape_model.graph.value_info) + list(
            inferred_shape_model.graph.input):
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
    """
    This function generates subgraph that represents one direction of unrolled RNN layer

    Parameters:
      transformer (ModelTransformerHelper): helper for model generation
      X (list of str): names of input tensors in sequence. Tensor shapes: [batch_size, input_size].
      W (list of str): name of weight tensor
      R (list of str): name of recurrence weight tensor
      B (list of str): name of bias tensor
      initial_h (str or None): name of tensor containing initial hidden state. Shape [batch_size, hidden_size]
      clip (float or None): range which clips input of activations
      act (str): activation function
    """
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
            state_tensor = transformer.make_node(
                'Clip', state_tensor, 1, min=-clip, max=clip)
        previous_state_tensor = transformer.make_node(activation, state_tensor, 1)
        state_tensors += previous_state_tensor

    for i in range(first_iter, seq_length):
        state_tensor = transformer.make_node('Gemm', [X[i]] + W + B, 1, transB=1)
        state_tensor = transformer.make_node(
            'Gemm', previous_state_tensor + R + state_tensor, 1, transB=1)
        if clip != None:
            state_tensor = transformer.make_node(
                'Clip', state_tensor, 1, min=-clip, max=clip)
        previous_state_tensor = transformer.make_node(activation, state_tensor, 1)
        state_tensors += previous_state_tensor
    return state_tensors


def transform_unidirectional_RNN(transformer, original_node, x, tensor_infos, activations,
                                 clip, direction, hidden_size, layout):
    inputs = original_node.input
    outputs = original_node.output
    if direction == 'reverse':
        x.reverse()
    w = transformer.make_node('Squeeze', [inputs[1]], 1, axes=[0])
    r = transformer.make_node('Squeeze', [inputs[2]], 1, axes=[0])
    if len(inputs) > 3 and inputs[3] != '':
        raw_bias_tensor = transformer.make_node('Squeeze', [inputs[3]], 1, axes=[0])
        splitted_bias_tensors = transformer.make_node(
            'Split', raw_bias_tensor, 2, axis=0, split=[hidden_size] * 2)
        b = transformer.make_node('Add', splitted_bias_tensors, 1)
    else:
        data_type = dtype_to_np(tensor_infos[inputs[2]].dtype)
        b = transformer.make_constant_tensor(
            np.zeros(hidden_size, dtype=data_type), "zero_bias")
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_node('Squeeze', [inputs[5]], 1, axes=[direction_dim])
    else:
        initial_h = None
    activation = activations[0]
    state_tensors = generate_one_direction_RNN(transformer, x, w, r, b, initial_h, clip,
                                               activation)
    y_direction_dim = layout + 1
    y_h_direction_dim = layout
    state_layout_tensors = []
    seq_length_dim = layout
    for state in state_tensors:
        state_layout_tensors += transformer.make_node(
            "Unsqueeze", [state], 1, axes=[seq_length_dim, y_direction_dim])

    Y_h = transformer.make_node(
        'Unsqueeze', [state_tensors[-1]], [outputs[1]], axes=[y_h_direction_dim])
    Y = transformer.make_node(
        'Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)


def transform_bidirectional_RNN(transformer, original_node, x, tensor_infos, activations,
                                clip, hidden_size, layout):
    inputs = original_node.input
    outputs = original_node.output
    w_bi = transformer.make_node('Split', [inputs[1]], 2, axis=0, split=[1, 1])
    r_bi = transformer.make_node('Split', [inputs[2]], 2, axis=0, split=[1, 1])
    w = []
    r = []
    for d in range(2):
        w += transformer.make_node('Squeeze', [w_bi[d]], 1, axes=[0])
        r += transformer.make_node('Squeeze', [r_bi[d]], 1, axes=[0])

    b = []
    if len(inputs) > 3 and inputs[3] != '':
        raw_bias_tensors = transformer.make_node(
            'Split', [inputs[3]], 2, axis=0, split=[1, 1])
        for d in range(2):
            raw_bias_tensors_squeezed = transformer.make_node(
                'Squeeze', [raw_bias_tensors[d]], 1, axes=[0])
            splitted_bias_tensors = transformer.make_node(
                'Split', raw_bias_tensors_squeezed, 2, axis=0, split=[hidden_size] * 2)
            b += transformer.make_node('Add', splitted_bias_tensors, 1)
    else:
        data_type = dtype_to_np(tensor_infos[inputs[2]].dtype)
        b = transformer.make_constant_tensor(
            np.zeros(hidden_size, dtype=data_type), "zero_bias") * 2
    initial_h = [None, None]
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_node(
            'Split', [inputs[5]], 2, axis=direction_dim, split=[1, 1])
        for d in range(2):
            initial_h[d] = transformer.make_node(
                'Squeeze', [initial_h[d]], 1, axes=[direction_dim])

    state_f_tensors = generate_one_direction_RNN(transformer, x, [w[0]], [r[0]], [b[0]],
                                                 initial_h[0], clip, activations[0])
    x.reverse()
    state_b_tensors = generate_one_direction_RNN(transformer, x, [w[1]], [r[1]], [b[1]],
                                                 initial_h[1], clip, activations[1])
    state_b_tensors.reverse()

    y_direction_dim = layout + 1
    y_h_direction_dim = layout
    state_layout_tensors = []
    seq_length_dim = layout
    seq_length = len(x)
    for t in range(seq_length):
        state_f = state_f_tensors[t]
        state_b = state_b_tensors[t]
        state_layout_tensors_f = transformer.make_node(
            "Unsqueeze", [state_f], 1, axes=[seq_length_dim, y_direction_dim])
        state_layout_tensors_b = transformer.make_node(
            "Unsqueeze", [state_b], 1, axes=[seq_length_dim, y_direction_dim])
        state_layout_tensors += transformer.make_node(
            "Concat",
            state_layout_tensors_f + state_layout_tensors_b,
            1,
            axis=y_direction_dim)

    last_f_state_layout_tensor = transformer.make_node(
        "Unsqueeze", [state_f_tensors[-1]], 1, axes=[y_h_direction_dim])
    last_b_state_layout_tensor = transformer.make_node(
        "Unsqueeze", [state_b_tensors[0]], 1, axes=[y_h_direction_dim])
    Y_h = transformer.make_node(
        'Concat',
        last_f_state_layout_tensor + last_b_state_layout_tensor, [outputs[1]],
        axis=y_h_direction_dim)

    Y = transformer.make_node(
        'Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)


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

    input_split_tensor = transformer.make_node(
        'Split', [inputs[0]], seq_length, axis=seq_length_dim, split=[1] * seq_length)
    x = []
    for i in range(len(input_split_tensor)):
        input_frame_tensor = input_split_tensor[i]
        squeezed_frame_tensor = transformer.make_node(
            'Squeeze', [input_frame_tensor], 1, axes=[0])
        x += squeezed_frame_tensor

    if direction in ['forward', 'reverse']:
        transform_unidirectional_RNN(transformer, node, x, tensor_infos, activations,
                                     clip, direction, hidden_size, layout)
    elif direction == 'bidirectional':
        transform_bidirectional_RNN(transformer, node, x, tensor_infos, activations, clip,
                                    hidden_size, layout)
    else:
        raise RuntimeError('Unknown RNN type')

    transformer.mark_for_deletion(node)


def generate_one_direction_LSTM(transformer, X, W, R, B, initial_h, initial_c, P, clip,
                                act, dtype, hidden_size, batch_size):
    """
    This function generates subgraph that represents one direction of unrolled LSTM layer

    Parameters:
        transformer (ModelTransformerHelper): helper for model generation
        tensor_infos (dict of Info): shapes and dtypes of tensors
        X (list of str): names of input tensors in sequence. Tensor shapes: [batch_size, input_size]
        W (list of str): name of concatenated weight tensor: [input, output, forget, cell]
        R (list of str): name of concatenated recurrence weights tensor: [input, output, forget, cell]
        B (list of str): name of concatenated bias tensor: [input, output, forget, cell]
        initial_h (str or None): name of tensor containing initial hidden state. Shape [batch_size, hidden_size]
        initial_c (str or None): name of tensor containing initial cell state. Shape [batch_size, hidden_size]
        P (list of str): name of concatenated peephole tensor: [input, output, forget]
        clip (float or None): range which clips input of activations
        act (dict of str):  activation functions {'f': 'Sigmoid', 'g': 'Tanh', 'h': 'Tanh'}
        hidden_size (int): hidden dimension
        batch_size (int): batch dimension
    """
    # one direction LSTM:
    #
    # From onnx Operators.onnx
    #
    # it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    # ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    # ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    # Ct = ft (.) Ct-1 + it (.) ct
    # ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    # Ht = ot (.) h(Ct)
    #
    # X - input tensor
    # i - input gate
    # o - output gate
    # f - forget gate
    # c - cell gate
    # t - time step (t-1 means previous time step)
    # W[iofc] - W parameter weight matrix for input, output, forget, and cell gates
    # R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates
    # Wb[iofc] - W bias vectors for input, output, forget, and cell gates
    # Rb[iofc] - R bias vectors for input, output, forget, and cell gates
    # P[iof] - P peephole weight vector for input, output, and forget gates
    # WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates
    # RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates
    # WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates
    # RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates
    # PB[iof] - P peephole weight vector for backward input, output, and forget gates
    # H - Hidden state

    seq_length = len(X)
    state_h_tensors = []

    w_tensors = transformer.make_node('Split', W, 4, axis=0, split=[hidden_size] * 4)
    W = {'i': w_tensors[0], 'o': w_tensors[1], 'f': w_tensors[2], 'c': w_tensors[3]}

    r_tensors = transformer.make_node('Split', R, 4, axis=0, split=[hidden_size] * 4)
    R = {'i': r_tensors[0], 'o': r_tensors[1], 'f': r_tensors[2], 'c': r_tensors[3]}

    if B is not None:
        separate_b_tensors = transformer.make_node(
            'Split', B, 8, axis=0, split=[hidden_size] * 8)
        b_tensors = []
        for i in range(4):
            b_tensors += transformer.make_node(
                'Add', [separate_b_tensors[i], separate_b_tensors[i + 4]], 1)
    else:
        b_tensors = transformer.make_constant_tensor(
            np.zeros((hidden_size), dtype=dtype), 'zero_b') * 4
    B = {'i': b_tensors[0], 'o': b_tensors[1], 'f': b_tensors[2], 'c': b_tensors[3]}

    if initial_h is not None:
        previous_h_state_tensor = initial_h
    else:
        previous_h_state_tensor = transformer.make_constant_tensor(
            np.zeros((batch_size, hidden_size), dtype=dtype), 'initial_h')[0]

    if initial_c is not None:
        previous_c_state_tensor = initial_c
    else:
        previous_c_state_tensor = transformer.make_constant_tensor(
            np.zeros((batch_size, hidden_size), dtype=dtype), 'initial_c')[0]

    if P is not None:
        p_tensors = transformer.make_node('Split', P, 3, axis=0, split=[hidden_size] * 3)
        P = {'i': p_tensors[0], 'o': p_tensors[1], 'f': p_tensors[2]}
    else:
        zero = transformer.make_constant_tensor(
            np.zeros((hidden_size), dtype=dtype), 'zero_peephole')[0]
        P = {'i': zero, 'o': zero, 'f': zero}

    for i in range(seq_length):
        # it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        it = transformer.make_node('Gemm', [X[i], W['i'], B['i']], 1, transB=1)
        it = transformer.make_node(
            'Gemm', [previous_h_state_tensor, R['i'], it[0]], 1, transB=1)
        peephole_it = transformer.make_node('Mul', [P['i'], previous_c_state_tensor], 1)
        it = transformer.make_node('Add', it + peephole_it, 1)
        if clip is not None:
            it = transformer.make_node('Clip', it, 1, min=-clip, max=clip)
        it = transformer.make_node(act['f'], it, 1)

        # ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        ft = transformer.make_node('Gemm', [X[i], W['f'], B['f']], 1, transB=1)
        ft = transformer.make_node(
            'Gemm', [previous_h_state_tensor, R['f'], ft[0]], 1, transB=1)
        peephole_ft = transformer.make_node('Mul', [P['f'], previous_c_state_tensor], 1)
        ft = transformer.make_node('Add', ft + peephole_ft, 1)
        if clip is not None:
            ft = transformer.make_node('Clip', ft, 1, min=-clip, max=clip)
        ft = transformer.make_node(act['f'], ft, 1)

        # ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        ct = transformer.make_node('Gemm', [X[i], W['c'], B['c']], 1, transB=1)
        ct = transformer.make_node(
            'Gemm', [previous_h_state_tensor, R['c'], ct[0]], 1, transB=1)
        if clip is not None:
            ct = transformer.make_node('Clip', ct, 1, min=-clip, max=clip)
        ct = transformer.make_node(act['g'], ct, 1)

        # Ct = ft (.) Ct-1 + it (.) ct
        ft_Ct = transformer.make_node('Mul', ft + [previous_c_state_tensor], 1)
        it_ct = transformer.make_node('Mul', it + ct, 1)
        Ct = transformer.make_node('Add', ft_Ct + it_ct, 1)
        previous_c_state_tensor = Ct[0]

        # ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        ot = transformer.make_node('Gemm', [X[i], W['o'], B['o']], 1, transB=1)
        ot = transformer.make_node(
            'Gemm', [previous_h_state_tensor, R['o'], ot[0]], 1, transB=1)
        peephole_ot = transformer.make_node('Mul', [P['o'], Ct[0]], 1)
        ot = transformer.make_node('Add', ot + peephole_ot, 1)
        if clip is not None:
            ot = transformer.make_node('Clip', ot, 1, min=-clip, max=clip)
        ot = transformer.make_node(act['f'], ot, 1)

        # Ht = ot (.) h(Ct)
        Ht = transformer.make_node(act['h'], Ct, 1)
        Ht = transformer.make_node('Mul', ot + Ht, 1)
        previous_h_state_tensor = Ht[0]
        state_h_tensors += Ht

    return (state_h_tensors, previous_c_state_tensor)


def transform_unidirectional_LSTM(transformer, original_node, x, tensor_infos,
                                  activations, clip, direction, hidden_size, layout):
    inputs = original_node.input
    outputs = original_node.output
    if direction == 'reverse':
        x.reverse()
    w = transformer.make_node('Squeeze', [inputs[1]], 1, axes=[0])
    r = transformer.make_node('Squeeze', [inputs[2]], 1, axes=[0])

    b = None
    if len(inputs) > 3 and inputs[3] != '':
        b = transformer.make_node('Squeeze', [inputs[3]], 1, axes=[0])

    initial_h = None
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_node(
            'Squeeze', [inputs[5]], 1, axes=[direction_dim])[0]

    initial_c = None
    if len(inputs) > 6 and inputs[6] != '':
        direction_dim = layout
        initial_c = transformer.make_node(
            'Squeeze', [inputs[6]], 1, axes=[direction_dim])[0]

    p = None
    if len(inputs) > 7 and inputs[7] != '':
        p = transformer.make_node('Squeeze', [inputs[7]], 1, axes=[0])

    dtype = dtype_to_np(tensor_infos[inputs[0]].dtype)
    batch_size = tensor_infos[inputs[0]].shape[1 - layout]

    act = {'f': activations[0], 'g': activations[1], 'h': activations[2]}

    state_h_tensors, state_c_tensor = generate_one_direction_LSTM(
        transformer, x, w, r, b, initial_h, initial_c, p, clip, act, dtype, hidden_size,
        batch_size)

    y_direction_dim = layout + 1
    y_h_direction_dim = layout
    state_layout_tensors = []
    seq_length_dim = layout
    for h_state in state_h_tensors:
        state_layout_tensors += transformer.make_node(
            "Unsqueeze", [h_state], 1, axes=[seq_length_dim, y_direction_dim])

    Y_h = transformer.make_node(
        'Unsqueeze', [state_h_tensors[-1]], [outputs[1]], axes=[y_h_direction_dim])
    Y_c = transformer.make_node(
        'Unsqueeze', [state_c_tensor], [outputs[2]], axes=[y_h_direction_dim])
    if direction == 'reverse':
        state_layout_tensors.reverse()
    Y = transformer.make_node(
        'Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)


def transform_bidirectional_LSTM(transformer, original_node, x, tensor_infos, activations,
                                 clip, hidden_size, layout):
    inputs = original_node.input
    outputs = original_node.output

    w = transformer.make_node('Split', [inputs[1]], 2, axis=0, split=[1, 1])
    r = transformer.make_node('Split', [inputs[2]], 2, axis=0, split=[1, 1])
    for d in range(2):
        w[d] = transformer.make_node('Squeeze', [w[d]], 1, axes=[0])
        r[d] = transformer.make_node('Squeeze', [r[d]], 1, axes=[0])

    b = [None, None]
    if len(inputs) > 3 and inputs[3] != '':
        b = transformer.make_node('Split', [inputs[3]], 2, axis=0, split=[1, 1])
        for d in range(2):
            b[d] = transformer.make_node('Squeeze', [b[d]], 1, axes=[0])

    initial_h = [None, None]
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_node(
            'Split', [inputs[5]], 2, axis=direction_dim, split=[1, 1])
        for d in range(2):
            initial_h[d] = transformer.make_node(
                'Squeeze', [initial_h[d]], 1, axes=[direction_dim])[0]

    initial_c = [None, None]
    if len(inputs) > 6 and inputs[6] != '':
        direction_dim = layout
        initial_c = transformer.make_node(
            'Split', [inputs[6]], 2, axis=direction_dim, split=[1, 1])
        for d in range(2):
            initial_c[d] = transformer.make_node(
                'Squeeze', [initial_c[d]], 1, axes=[direction_dim])[0]

    p = [None, None]
    if len(inputs) > 7 and inputs[7] != '':
        p = transformer.make_node('Split', [inputs[7]], 2, axis=0, split=[1, 1])
        for d in range(2):
            p[d] = transformer.make_node('Squeeze', [p[d]], 1, axes=[0])

    dtype = dtype_to_np(tensor_infos[inputs[0]].dtype)
    batch_size = tensor_infos[inputs[0]].shape[1 - layout]

    act = [{
        'f': activations[0],
        'g': activations[1],
        'h': activations[2]
    }, {
        'f': activations[3],
        'g': activations[4],
        'h': activations[5]
    }]

    state_f_h_tensors, state_f_c_tensor = generate_one_direction_LSTM(
        transformer, x, w[0], r[0], b[0], initial_h[0], initial_c[0], p[0], clip, act[0],
        dtype, hidden_size, batch_size)
    x.reverse()
    state_b_h_tensors, state_b_c_tensor = generate_one_direction_LSTM(
        transformer, x, w[1], r[1], b[1], initial_h[1], initial_c[1], p[1], clip, act[1],
        dtype, hidden_size, batch_size)
    state_b_h_tensors.reverse()

    y_direction_dim = layout + 1
    y_c_direction_dim = layout
    state_layout_tensors = []
    seq_length_dim = layout
    for f_h_state, b_h_state in zip(state_f_h_tensors, state_b_h_tensors):
        state_f_layout_tensors = transformer.make_node(
            "Unsqueeze", [f_h_state], 1, axes=[seq_length_dim, y_direction_dim])
        state_b_layout_tensors = transformer.make_node(
            "Unsqueeze", [b_h_state], 1, axes=[seq_length_dim, y_direction_dim])
        state_layout_tensors += transformer.make_node(
            'Concat',
            state_f_layout_tensors + state_b_layout_tensors,
            1,
            axis=y_direction_dim)

    last_f_state_layout_tensor = transformer.make_node(
        "Unsqueeze", [state_f_h_tensors[-1]], 1, axes=[y_c_direction_dim])
    last_b_state_layout_tensor = transformer.make_node(
        "Unsqueeze", [state_b_h_tensors[0]], 1, axes=[y_c_direction_dim])
    Y_h = transformer.make_node(
        'Concat',
        last_f_state_layout_tensor + last_b_state_layout_tensor, [outputs[1]],
        axis=y_c_direction_dim)

    Y_f_c = transformer.make_node(
        'Unsqueeze', [state_f_c_tensor], 1, axes=[y_c_direction_dim])
    Y_b_c = transformer.make_node(
        'Unsqueeze', [state_b_c_tensor], 1, axes=[y_c_direction_dim])
    Y_c = transformer.make_node(
        'Concat', Y_f_c + Y_b_c, [outputs[2]], axis=y_c_direction_dim)

    Y = transformer.make_node(
        'Concat', state_layout_tensors, [outputs[0]], axis=seq_length_dim)


def legalize_LSTM(transformer, tensor_infos, node):
    inputs = node.input
    outputs = node.output
    if len(inputs) > 4 and inputs[4] != '':
        raise NotImplementedError('Variadic length of output is not supported')
    name = node.name
    # attributes
    activation_alpha = []
    activation_beta = []
    activations = ['Sigmoid', 'Tanh', 'Tanh'] * 2
    clip = None
    direction = 'forward'
    hidden_size = 0
    input_forget = 0
    layout = 0

    for attr in node.attribute:
        print("!!! note.addtribute", attr)

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
        if attr.name == 'input_forget':
            input_forget = attr.i
        if attr.name == 'layout':
            layout = attr.i

    for act in activations:
        if act not in ['Relu', 'Tanh', 'Sigmoid']:
            raise NotImplementedError('Unsupported activation function')

    if input_forget != 0:
        raise NotImplementedError('Unsupported input_forget attribute value')

    seq_length_dim = layout
    seq_length = tensor_infos[inputs[0]].shape[seq_length_dim]
    if hidden_size == 0:
        hidden_size = tensor_infos[inputs[2]].shape[2]

    seq_length = 1

    input_split_tensor = transformer.make_node(
        'Split', [inputs[0]], seq_length, axis=seq_length_dim, split=[1] * seq_length)
    x = []
    for i in range(len(input_split_tensor)):
        input_frame_tensor = input_split_tensor[i]
        squeezed_frame_tensor = transformer.make_node(
            'Squeeze', [input_frame_tensor], 1, axes=[0])
        x += squeezed_frame_tensor

    if direction in ['forward', 'reverse']:
        transform_unidirectional_LSTM(transformer, node, x, tensor_infos, activations,
                                      clip, direction, hidden_size, layout)
    elif direction == 'bidirectional':
        transform_bidirectional_LSTM(transformer, node, x, tensor_infos, activations,
                                     clip, hidden_size, layout)
    else:
        raise RuntimeError('Unknown LSTM type')

    transformer.mark_for_deletion(node)


def legalize(model):
    tensor_infos = get_tensor_infos(model)

    transformer = ModelTransformerHelper(model)

    node_id = 0
    while node_id < len(model.graph.node):
        node = model.graph.node[node_id]
        if node.op_type == 'RNN':
            # opset version is required by split operation
            if model.opset_import[0].version >= 13:
                raise NotImplementedError(
                    'Can not generate code with opcode version 13 and greater')
            transformer.set_insert_id(node_id)
            legalize_RNN(transformer, tensor_infos, node)
            node_id = transformer.get_insert_id()
        elif node.op_type == 'LSTM':
            if model.opset_import[0].version >= 13:
                raise NotImplementedError(
                    'Can not generate code with opcode version 13 and greater')
            transformer.set_insert_id(node_id)
            legalize_LSTM(transformer, tensor_infos, node)
            node_id = transformer.get_insert_id()
        node_id += 1

    transformer.delete_marked_nodes()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: ./legalize_onnx.py <path to input model> <path to output model>')
        exit(1)
    model = onnx.load(sys.argv[1])
    legalize(model)
    onnx.save(model, sys.argv[2])
