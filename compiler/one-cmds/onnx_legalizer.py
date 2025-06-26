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
#
# This code works with onnx model in proto format. See proto buffers format in
# https://github.com/onnx/onnx/blob/96516aecd4c110b0ac57eba08ac236ebf7205728/onnx/onnx.proto3
#
# More examples of handling onnx models could be found here:
# https://github.com/onnx/onnx/tree/96516aecd4c110b0ac57eba08ac236ebf7205728/onnx/examples
#
# List of transformations:
# - Replace RNN operation with unrolled subgraph
# - Replace LSTM operation with unrolled subgraph


class LegalizeOptions:
    """Controls transformations that legalizer apply

    Attributes:
        unroll_rnn (bool): default is False. If True - unrolls RNN operations
        unroll_lstm (bool): default is False. If True - unrolls LSTM operations
    """

    unroll_rnn = False
    unroll_lstm = False


def _reverse_str(s):
    return ''.join(reversed(s))


def _parse_tensor_name(name):
    """Splits tensor name to base part and serial number

    Most of tensor names have following format: "tensor_123".
    This  function breaks name into two values: "tensor_" and 123.
    Tensor names like this: "321" are broken into "" and 321.

    Serial number is used to create unique tensor names using given base name.

    Args:
        name (str): tensor name

    Returns:
        tuple of str, int: base name and serial number of tensor
    """
    rev = _reverse_str(name)
    m = re.match(r'(\d*)(.*)', rev)
    if m.groups()[0] != '':
        return (_reverse_str(m.groups()[1]), int(_reverse_str(m.groups()[0])))
    else:
        return (_reverse_str(m.groups()[1]), 0)


class _ModelTransformerHelper:
    """Helper for onnx model transformation

    This helper is used for convenient operation replacement in onnx model

    Attributes:
        _model (onnx.onnx_ml_pb2.ModelProto): target model that should be changed
        _nodes_to_delete (list of onnx.onnx_ml_pb2.NodeProto): list of replaced operations
        _insert_id (int): position to insert created operations (should be in topologically sorted)
        _base_name_idx (dict from str to int): maps tensor "base" name to
            largest existing serial num. For example model has tensors "t_1", "t_2", "t_4",
            in that case _base_name_idx["t_"] == 4.
            This attribute is used for unique tensor name generation.
    """
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
                base_name, number = _parse_tensor_name(t)
                if base_name in self._base_name_idx:
                    self._base_name_idx[base_name] = max(self._base_name_idx[base_name],
                                                         number)
                else:
                    self._base_name_idx[base_name] = number

    def make_tensor_with_base_name(self, base_name):
        """ Create unique name for given base_name

        Args:
            base_name (str): base tensor name

        Returns:
            str : unique tensor name that starts with base_name
        """
        if base_name in self._base_name_idx:
            self._base_name_idx[base_name] += 1
            return base_name + str(self._base_name_idx[base_name])
        else:
            self._base_name_idx[base_name] = 0
            return base_name + '0'

    def make_node(self, opcode, inputs, outputs, *p_args, **k_args):
        """Create arbitrary node and insert it in graph.

        Args:
            opcode (str): opcode name of desired operation
            inputs (list of str): names of input tensors
            outputs (list of str or int): names of existing tensors to use as output tensors for operation or
                number of tensors that should be created
            p_args: additional arguments for onnx make_node helper
            k_args: attributes for onnx node

        Returns:
            list of str: list of output tensor names
        """
        if type(outputs) == int:
            outputs = [self.make_tensor_with_base_name('') for i in range(outputs)]
        assert (type(outputs) == list)
        node = onnx.helper.make_node(opcode, inputs, outputs, *p_args, **k_args)
        self._model.graph.node.insert(self._insert_id, node)
        self._insert_id += 1
        return outputs

    def make_split(self, input, split_sizes, axis):
        """Create Split operation and insert it in graph.

        Args:
            input (str): name of input tensor
            split_sizes (list of int): list of split sizes
            axis (int): number of axis to split

        Returns:
            list: list of output tensor names
        """
        return self.make_node('Split', [input],
                              len(split_sizes),
                              axis=axis,
                              split=split_sizes)

    def make_concat(self, inputs, axis):
        """Create Concat operation and insert it in graph.

        Args:
            inputs (list of str): list of tensors names to concat
            axis (int): axis number to concat

        Returns:
            str: output tensor name
        """
        return self.make_node('Concat', inputs, 1, axis=axis)[0]

    def make_squeeze(self, input, axes):
        """Create Squeeze operation and insert it in graph.

        Args:
            input (str): name of input tensor
            axes (list of int): list of dimension containing ones to remove

        Returns:
            str: output tensor name
        """
        return self.make_node('Squeeze', [input], 1, axes=axes)[0]

    def make_unsqueeze(self, input, axes):
        """Create Unsqueeze operation and insert it in graph.

        Args:
            input (str): name of input tensor
            axes (list of int): list of dimension to insert ones

        Returns:
            str: output tensor name
        """
        return self.make_node('Unsqueeze', [input], 1, axes=axes)[0]

    def make_gemm(self, A, B, C, trans_a=False, trans_b=False):
        """Create Gemm operation and insert it in graph.

        Result tensor contains A*B + C

        Args:
            A (str): name of tensor A
            B (str): name of tensor B
            C (str): name of tensor C
            transA (bool): if True, transpose tensor A before multiplication
            transB (bool): if True, transpose tensor B before multiplication

        Returns:
            str: output tensor name
        """
        return self.make_node('Gemm', [A, B, C],
                              1,
                              transA=bool(trans_a),
                              transB=bool(trans_b))[0]

    def make_add(self, a, b):
        """Creates Add operation and insert it in graph.

        Args:
            a (str): name of left operand tensor
            b (str): name of right operand tensor

        Returns:
            str: output tensor name
        """
        return self.make_node('Add', [a, b], 1)[0]

    def make_mul(self, a, b):
        """Creates Mul operation and insert it in graph.

        Args:
            a (str): name of left operand tensor
            b (str): name of right operand tensor

        Returns:
            str: output tensor name
        """
        return self.make_node('Mul', [a, b], 1)[0]

    def make_clip(self, input, min, max):
        """Create Clip operation and insert it in graph.

        Args:
            input (str): input tensor name
            min (int/float): lower clip bound
            max (int/float ): upper clip bound

        Returns:
            str: output tensor name
        """
        return self.make_node('Clip', [input], 1, min=min, max=max)[0]

    def make_act(self, input, act_name):
        """Create activation function operation and insert it in graph.

        Args:
            input (str): input tensor name
            act_name (str): name of activation function, one of ['Relu', 'Tanh', 'Sigmoid']

        Returns:
            str: output tensor name
        """
        assert (act_name in ['Relu', 'Tanh', 'Sigmoid'])
        return self.make_node(act_name, [input], 1)[0]

    def make_constant_tensor(self, tensor_data, base_name):
        """Creates onnx constant tensor

        Args:
            tensor_data (numpy.ndarray): tensor data
            base_name (str): prefix of constant tensor name

        Returns:
            str: name of created constant tensor
        """
        tensor = onnx.numpy_helper.from_array(tensor_data)
        tensor.name = self.make_tensor_with_base_name(base_name)
        self._model.graph.initializer.append(tensor)
        return tensor.name

    def mark_for_deletion(self, node):
        self._nodes_to_delete += [node]

    def get_insert_id(self):
        return self._insert_id

    def set_insert_id(self, insert_id):
        self._insert_id = insert_id

    def delete_marked_nodes(self):
        for node in self._nodes_to_delete:
            self._model.graph.node.remove(node)


class _TensorInfo:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape


def _get_tensor_infos(model):
    """Infer tensor shapes and dtypes
    Args:
        model (onnx.onnx_ml_pb2.ModelProto): model to process

    Returns:
        dict from str to _TensorInfo: maps tensor name to shape and dtype information
    """

    inferred_shape_model = onnx.shape_inference.infer_shapes(model)

    infos = {}
    for tensor in list(inferred_shape_model.graph.value_info) + list(
            inferred_shape_model.graph.input):
        info = _TensorInfo(tensor.type.tensor_type.elem_type, [])
        for dim in tensor.type.tensor_type.shape.dim:
            info.shape += [dim.dim_value]
        infos[tensor.name] = info

    for tensor in list(model.graph.initializer):
        infos[tensor.name] = _TensorInfo(tensor.data_type, tensor.dims)
    return infos


def _dtype_to_np(dtype):
    """Convert onnx dtype value to numpy dtype class

    For more types see:
    https://github.com/onnx/onnx/blob/96516aecd4c110b0ac57eba08ac236ebf7205728/onnx/onnx.proto3#L484

    Args:
        dtype (int): onnx dtype

    Returns:
        numpy data type: numpy dtype, like np.float32
    """

    if dtype == 1:
        return np.float32
    else:
        raise NotImplementedError('unsupported data type')


def _generate_one_direction_RNN(transformer, X, W, R, B, initial_h, clip,
                                activation_name):
    """Generate subgraph of one direction of unrolled RNN layer

    Args:
        transformer (_ModelTransformerHelper): helper for model generation
        X (list of str): names of input tensors in sequence. Tensor shapes: [batch_size, input_size].
        W (str): name of weight tensor
        R (str): name of recurrence weight tensor
        B (str): name of bias tensor
        initial_h (str or None): name of tensor containing initial hidden state. Shape [batch_size, hidden_size]
        clip (float or None): range which clips input of activations
        act (str): activation function
    """
    # one direction RNN:
    #
    # For details see:
    # https://github.com/onnx/onnx/blob/5cf5feef5ec3fd5527b2fdb6c29780e3b705059f/docs/Changelog.md#RNN-7
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
        state_tensor = transformer.make_gemm(X[0], W, B, trans_b=True)
        if clip != None:
            state_tensor = transformer.make_clip(state_tensor, min=-clip, max=clip)
        previous_state_tensor = transformer.make_act(state_tensor, activation_name)
        state_tensors += [previous_state_tensor]

    for i in range(first_iter, seq_length):
        state_tensor = transformer.make_gemm(X[i], W, B, trans_b=True)
        state_tensor = transformer.make_gemm(previous_state_tensor,
                                             R,
                                             state_tensor,
                                             trans_b=True)
        if clip != None:
            state_tensor = transformer.make_clip(state_tensor, min=-clip, max=clip)
        previous_state_tensor = transformer.make_act(state_tensor, activation_name)
        state_tensors += [previous_state_tensor]
    return state_tensors


def _transform_unidirectional_RNN(transformer, original_node, x, tensor_infos, activation,
                                  clip, direction, hidden_size, layout):
    """Generate Simple (forward or reverse) unrolled RNN

    Args:
        transformer (_ModelTransformerHelper): transformation helper
        original_node (onnx.onnx_ml_pb2.NodeProto): unidirectional RNN operation to unroll
        x (list of str): list of input tensors (input tensor split along "time" dimension)
        tensor_infos (dict from str to _TensorInfo): dict maps tensor name to it's shape and dtype info
        activation (str): name of activation function
        clip (float or None): range which clips input of activations
        direction (str): "forward" or "reverse"
        hidden_size (int): size of hidden state
        layout (int): See attribute description:
            https://github.com/onnx/onnx/blob/5cf5feef5ec3fd5527b2fdb6c29780e3b705059f/docs/Operators.md#attributes-56
    """

    inputs = original_node.input
    outputs = original_node.output
    if direction == 'reverse':
        x.reverse()
    w = transformer.make_squeeze(inputs[1], axes=[0])
    r = transformer.make_squeeze(inputs[2], axes=[0])
    if len(inputs) > 3 and inputs[3] != '':
        raw_bias_tensor = transformer.make_squeeze(inputs[3], axes=[0])
        splitted_bias_tensors = transformer.make_split(raw_bias_tensor,
                                                       split_sizes=[hidden_size] * 2,
                                                       axis=0)
        b = transformer.make_add(splitted_bias_tensors[0], splitted_bias_tensors[1])
    else:
        data_type = _dtype_to_np(tensor_infos[inputs[2]].dtype)
        b = transformer.make_constant_tensor(np.zeros(hidden_size, dtype=data_type),
                                             "zero_bias")
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_squeeze(inputs[5], axes=[direction_dim])
    else:
        initial_h = None
    state_tensors = _generate_one_direction_RNN(transformer, x, w, r, b, initial_h, clip,
                                                activation)
    y_direction_dim = layout + 1
    y_h_direction_dim = layout
    state_layout_tensors = []
    seq_length_dim = layout
    for state in state_tensors:
        state_layout_tensors += [
            transformer.make_unsqueeze(state, axes=[seq_length_dim, y_direction_dim])
        ]

    # use low-level interface to attach to existing tensors
    Y_h = outputs[1]
    transformer.make_node('Unsqueeze', [state_tensors[-1]], [Y_h],
                          axes=[y_h_direction_dim])
    Y = outputs[0]
    transformer.make_node('Concat', state_layout_tensors, [Y], axis=seq_length_dim)


def _transform_bidirectional_RNN(transformer, original_node, x, tensor_infos, activations,
                                 clip, hidden_size, layout):
    """Generate Bidirectional unrolled RNN

    Args:
        transformer (_ModelTransformerHelper): transformation helper
        original_node (onnx.onnx_ml_pb2.NodeProto): bidirectional RNN operation to unroll
        x (list of str): list of input tensors (input tensor split along "time" dimension)
        tensor_infos (dict from str to _TensorInfo): dict maps tensor name to it's shape and dtype info
        activations (list of str): list of len (2) containing names of forward and reverse activations
        clip (float or None): range which clips input of activations
        hidden_size (int): size of hidden state
        layout (int): See attribute description:
            https://github.com/onnx/onnx/blob/5cf5feef5ec3fd5527b2fdb6c29780e3b705059f/docs/Operators.md#attributes-56
    """

    inputs = original_node.input
    outputs = original_node.output
    w_bi = transformer.make_split(inputs[1], split_sizes=[1, 1], axis=0)
    r_bi = transformer.make_split(inputs[2], split_sizes=[1, 1], axis=0)
    w = []
    r = []
    for d in range(2):
        w += [transformer.make_squeeze(w_bi[d], axes=[0])]
        r += [transformer.make_squeeze(r_bi[d], axes=[0])]

    b = []
    if len(inputs) > 3 and inputs[3] != '':
        raw_bias_tensors = transformer.make_split(inputs[3], split_sizes=[1, 1], axis=0)
        for d in range(2):
            raw_bias_tensors_squeezed = transformer.make_squeeze(raw_bias_tensors[d],
                                                                 axes=[0])
            splitted_bias_tensors = transformer.make_split(raw_bias_tensors_squeezed,
                                                           split_sizes=[hidden_size] * 2,
                                                           axis=0)
            b += [
                transformer.make_add(splitted_bias_tensors[0], splitted_bias_tensors[1])
            ]
    else:
        data_type = _dtype_to_np(tensor_infos[inputs[2]].dtype)
        b = [
            transformer.make_constant_tensor(np.zeros(hidden_size, dtype=data_type),
                                             "zero_bias")
        ] * 2
    initial_h = [None, None]
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_split(inputs[5],
                                           split_sizes=[1, 1],
                                           axis=direction_dim)
        for d in range(2):
            initial_h[d] = transformer.make_squeeze(initial_h[d], axes=[direction_dim])

    state_f_tensors = _generate_one_direction_RNN(transformer, x, w[0], r[0], b[0],
                                                  initial_h[0], clip, activations[0])
    x.reverse()
    state_b_tensors = _generate_one_direction_RNN(transformer, x, w[1], r[1], b[1],
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
        state_layout_tensors_f = transformer.make_unsqueeze(
            state_f, axes=[seq_length_dim, y_direction_dim])
        state_layout_tensors_b = transformer.make_unsqueeze(
            state_b, axes=[seq_length_dim, y_direction_dim])
        state_layout_tensors += [
            transformer.make_concat([state_layout_tensors_f, state_layout_tensors_b],
                                    axis=y_direction_dim)
        ]

    last_f_state_layout_tensor = transformer.make_unsqueeze(state_f_tensors[-1],
                                                            axes=[y_h_direction_dim])
    last_b_state_layout_tensor = transformer.make_unsqueeze(state_b_tensors[0],
                                                            axes=[y_h_direction_dim])

    # use low-level interface to attach to existing tensors
    Y_h = outputs[1]
    transformer.make_node('Concat',
                          [last_f_state_layout_tensor, last_b_state_layout_tensor], [Y_h],
                          axis=y_h_direction_dim)

    Y = outputs[0]
    transformer.make_node('Concat', state_layout_tensors, [Y], axis=seq_length_dim)


def _legalize_RNN(transformer, tensor_infos, node):
    """Unroll RNN operation

    Args:
        transformer (_ModelTransformerHelper): transformation helper
        tensor_infos (dict from str to _TensorInfo): dict maps tensor name to it's shape and dtype info
        node (onnx.onnx_ml_pb2.NodeProto): RNN operation to unroll
    """
    inputs = node.input
    if len(inputs) > 4 and inputs[4] != '':
        raise NotImplementedError('Variadic length of output is not supported')
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

    if len(activation_alpha) > 0 or len(activation_beta) > 0:
        raise NotImplementedError('Unsupported parameters for LSTM activations')

    for act in activations:
        if act not in ['Relu', 'Tanh', 'Sigmoid']:
            raise NotImplementedError('Unsupported activation function')

    seq_length_dim = layout
    seq_length = tensor_infos[inputs[0]].shape[seq_length_dim]
    if hidden_size == 0:
        hidden_size = tensor_infos[inputs[2]].shape[2]

    input_split_tensor = transformer.make_split(inputs[0],
                                                split_sizes=[1] * seq_length,
                                                axis=seq_length_dim)
    x = []
    for i in range(len(input_split_tensor)):
        input_frame_tensor = input_split_tensor[i]
        squeezed_frame_tensor = transformer.make_squeeze(input_frame_tensor, axes=[0])
        x += [squeezed_frame_tensor]

    if direction in ['forward', 'reverse']:
        _transform_unidirectional_RNN(transformer, node, x, tensor_infos, activations[0],
                                      clip, direction, hidden_size, layout)
    elif direction == 'bidirectional':
        _transform_bidirectional_RNN(transformer, node, x, tensor_infos, activations,
                                     clip, hidden_size, layout)
    else:
        raise RuntimeError('Unknown RNN type')

    transformer.mark_for_deletion(node)


def _generate_one_direction_LSTM(transformer, X, W, R, B, initial_h, initial_c, P, clip,
                                 act, dtype, hidden_size, batch_size):
    """Generate subgraph for one direction of unrolled LSTM layer

    Args:
        transformer (_ModelTransformerHelper): helper for model generation
        X (list of str): names of tensors in input sequence. Each tensor shape: [batch_size, input_size]
        W (str): name of concatenated weight tensor: [input, output, forget, cell]
        R (str): name of concatenated recurrence weights tensor: [input, output, forget, cell]
        B (str): name of concatenated bias tensor: [input, output, forget, cell]
        initial_h (str or None): name of tensor containing initial hidden state. Shape [batch_size, hidden_size]
        initial_c (str or None): name of tensor containing initial cell state. Shape [batch_size, hidden_size]
        P (str or None): name of concatenated peephole tensor: [input, output, forget]
        clip (float or None): range which clips input of activations
        act (dict of str):  activation functions {'f': 'Sigmoid', 'g': 'Tanh', 'h': 'Tanh'}
        dtype (numpy dtype): data type used in created LSTM operation
        hidden_size (int): hidden dimension
        batch_size (int): batch dimension
    """
    # one direction LSTM:
    #
    # For details see:
    # https://github.com/onnx/onnx/blob/5cf5feef5ec3fd5527b2fdb6c29780e3b705059f/docs/Changelog.md#LSTM-7
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

    w_tensors = transformer.make_split(W, split_sizes=[hidden_size] * 4, axis=0)
    W = {'i': w_tensors[0], 'o': w_tensors[1], 'f': w_tensors[2], 'c': w_tensors[3]}

    r_tensors = transformer.make_split(R, split_sizes=[hidden_size] * 4, axis=0)
    R = {'i': r_tensors[0], 'o': r_tensors[1], 'f': r_tensors[2], 'c': r_tensors[3]}

    if B is not None:
        separate_b_tensors = transformer.make_split(B,
                                                    split_sizes=[hidden_size] * 8,
                                                    axis=0)
        b_tensors = []
        for i in range(4):
            b_tensors += [
                transformer.make_add(separate_b_tensors[i], separate_b_tensors[i + 4])
            ]
    else:
        b_tensors = [
            transformer.make_constant_tensor(np.zeros(
                (hidden_size), dtype=dtype), 'zero_b')
        ] * 4
    B = {'i': b_tensors[0], 'o': b_tensors[1], 'f': b_tensors[2], 'c': b_tensors[3]}

    if initial_h is not None:
        previous_h_state_tensor = initial_h
    else:
        previous_h_state_tensor = transformer.make_constant_tensor(
            np.zeros((batch_size, hidden_size), dtype=dtype), 'initial_h')

    if initial_c is not None:
        previous_c_state_tensor = initial_c
    else:
        previous_c_state_tensor = transformer.make_constant_tensor(
            np.zeros((batch_size, hidden_size), dtype=dtype), 'initial_c')

    if P is not None:
        p_tensors = transformer.make_split(P, split_sizes=[hidden_size] * 3, axis=0)
        P = {'i': p_tensors[0], 'o': p_tensors[1], 'f': p_tensors[2]}
    else:
        zero = transformer.make_constant_tensor(np.zeros((hidden_size), dtype=dtype),
                                                'zero_peephole')
        P = {'i': zero, 'o': zero, 'f': zero}

    for i in range(seq_length):
        # it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        it = transformer.make_gemm(X[i], W['i'], B['i'], trans_b=True)
        it = transformer.make_gemm(previous_h_state_tensor, R['i'], it, trans_b=True)
        peephole_it = transformer.make_mul(P['i'], previous_c_state_tensor)
        it = transformer.make_add(it, peephole_it)
        if clip is not None:
            it = transformer.make_clip(it, min=-clip, max=clip)
        it = transformer.make_act(it, act['f'])

        # ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        ft = transformer.make_gemm(X[i], W['f'], B['f'], trans_b=True)
        ft = transformer.make_gemm(previous_h_state_tensor, R['f'], ft, trans_b=True)
        peephole_ft = transformer.make_mul(P['f'], previous_c_state_tensor)
        ft = transformer.make_add(ft, peephole_ft)
        if clip is not None:
            ft = transformer.make_clip(ft, min=-clip, max=clip)
        ft = transformer.make_act(ft, act['f'])

        # ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        ct = transformer.make_gemm(X[i], W['c'], B['c'], trans_b=True)
        ct = transformer.make_gemm(previous_h_state_tensor, R['c'], ct, trans_b=True)
        if clip is not None:
            ct = transformer.make_clip(ct, min=-clip, max=clip)
        ct = transformer.make_act(ct, act['g'])

        # Ct = ft (.) Ct-1 + it (.) ct
        ft_Ct = transformer.make_mul(ft, previous_c_state_tensor)
        it_ct = transformer.make_mul(it, ct)
        Ct = transformer.make_add(ft_Ct, it_ct)
        previous_c_state_tensor = Ct

        # ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        ot = transformer.make_gemm(X[i], W['o'], B['o'], trans_b=True)
        ot = transformer.make_gemm(previous_h_state_tensor, R['o'], ot, trans_b=True)
        peephole_ot = transformer.make_mul(P['o'], Ct)
        ot = transformer.make_add(ot, peephole_ot)
        if clip is not None:
            ot = transformer.make_clip(ot, min=-clip, max=clip)
        ot = transformer.make_act(ot, act['f'])

        # Ht = ot (.) h(Ct)
        Ht = transformer.make_act(Ct, act['h'])
        Ht = transformer.make_mul(ot, Ht)
        previous_h_state_tensor = Ht
        state_h_tensors += [Ht]

    return (state_h_tensors, previous_c_state_tensor)


def _transform_unidirectional_LSTM(transformer, original_node, x, tensor_infos,
                                   activations, clip, direction, hidden_size, layout):
    """Generate Simple (forward or reverse) unrolled LSTM

    Args:
        transformer (_ModelTransformerHelper): transformation helper
        original_node (onnx.onnx_ml_pb2.NodeProto): unidirectional LSTM operation to unroll
        x (list of str): list of input tensors (input tensor split along "time" dimension)
        tensor_infos (dict from str to _TensorInfo): dict maps tensor name to it's shape and dtype info
        activations (list of str): list of length 3 containing names of activation functions
        clip (float or None): range which clips input of activations
        direction (str): "forward" or "reverse"
        hidden_size (int): size of hidden state
        layout (int): See attribute description:
            https://github.com/onnx/onnx/blob/5cf5feef5ec3fd5527b2fdb6c29780e3b705059f/docs/Operators.md#attributes-37
    """

    inputs = original_node.input
    outputs = original_node.output
    if direction == 'reverse':
        x.reverse()
    w = transformer.make_squeeze(inputs[1], axes=[0])
    r = transformer.make_squeeze(inputs[2], axes=[0])

    b = None
    if len(inputs) > 3 and inputs[3] != '':
        b = transformer.make_squeeze(inputs[3], axes=[0])

    initial_h = None
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_squeeze(inputs[5], axes=[direction_dim])

    initial_c = None
    if len(inputs) > 6 and inputs[6] != '':
        direction_dim = layout
        initial_c = transformer.make_squeeze(inputs[6], axes=[direction_dim])

    p = None
    if len(inputs) > 7 and inputs[7] != '':
        p = transformer.make_squeeze(inputs[7], axes=[0])

    dtype = _dtype_to_np(tensor_infos[inputs[0]].dtype)
    batch_size = tensor_infos[inputs[0]].shape[1 - layout]

    act = {'f': activations[0], 'g': activations[1], 'h': activations[2]}

    state_h_tensors, state_c_tensor = _generate_one_direction_LSTM(
        transformer, x, w, r, b, initial_h, initial_c, p, clip, act, dtype, hidden_size,
        batch_size)

    y_direction_dim = layout + 1
    y_h_direction_dim = layout
    state_layout_tensors = []
    seq_length_dim = layout
    for h_state in state_h_tensors:
        state_layout_tensors += [
            transformer.make_unsqueeze(h_state, axes=[seq_length_dim, y_direction_dim])
        ]

    # use low-level interface to attach to existing tensors
    Y_h = outputs[1]
    transformer.make_node('Unsqueeze', [state_h_tensors[-1]], [Y_h],
                          axes=[y_h_direction_dim])
    Y_c = outputs[2]
    transformer.make_node('Unsqueeze', [state_c_tensor], [Y_c], axes=[y_h_direction_dim])
    if direction == 'reverse':
        state_layout_tensors.reverse()
    Y = outputs[0]
    transformer.make_node('Concat', state_layout_tensors, [Y], axis=seq_length_dim)


def _transform_bidirectional_LSTM(transformer, original_node, x, tensor_infos,
                                  activations, clip, hidden_size, layout):
    """Generate Bidirectional unrolled LSTM

    Args:
        transformer (_ModelTransformerHelper): transformation helper
        original_node (onnx.onnx_ml_pb2.NodeProto): bidirectional LSTM operation to unroll
        x (list of str): list of input tensors (input tensor split along "time" dimension)
        tensor_infos (dict from str to _TensorInfo): dict maps tensor name to it's shape and dtype info
        activations (list of str): list of length 6, containing names of forward and reverse activations
        clip (float or None): range which clips input of activations
        hidden_size (int): size of hidden state
        layout (int): See attribute description:
            https://github.com/onnx/onnx/blob/5cf5feef5ec3fd5527b2fdb6c29780e3b705059f/docs/Operators.md#attributes-37
    """

    inputs = original_node.input
    outputs = original_node.output

    w = transformer.make_split(inputs[1], split_sizes=[1, 1], axis=0)
    r = transformer.make_split(inputs[2], split_sizes=[1, 1], axis=0)
    for d in range(2):
        w[d] = transformer.make_squeeze(w[d], axes=[0])
        r[d] = transformer.make_squeeze(r[d], axes=[0])

    b = [None, None]
    if len(inputs) > 3 and inputs[3] != '':
        b = transformer.make_split(inputs[3], split_sizes=[1, 1], axis=0)
        for d in range(2):
            b[d] = transformer.make_squeeze(b[d], axes=[0])

    initial_h = [None, None]
    if len(inputs) > 5 and inputs[5] != '':
        direction_dim = layout
        initial_h = transformer.make_split(inputs[5],
                                           split_sizes=[1, 1],
                                           axis=direction_dim)
        for d in range(2):
            initial_h[d] = transformer.make_squeeze(initial_h[d], axes=[direction_dim])

    initial_c = [None, None]
    if len(inputs) > 6 and inputs[6] != '':
        direction_dim = layout
        initial_c = transformer.make_split(inputs[6],
                                           split_sizes=[1, 1],
                                           axis=direction_dim)
        for d in range(2):
            initial_c[d] = transformer.make_squeeze(initial_c[d], axes=[direction_dim])

    p = [None, None]
    if len(inputs) > 7 and inputs[7] != '':
        p = transformer.make_split(inputs[7], split_sizes=[1, 1], axis=0)
        for d in range(2):
            p[d] = transformer.make_squeeze(p[d], axes=[0])

    dtype = _dtype_to_np(tensor_infos[inputs[0]].dtype)
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

    state_f_h_tensors, state_f_c_tensor = _generate_one_direction_LSTM(
        transformer, x, w[0], r[0], b[0], initial_h[0], initial_c[0], p[0], clip, act[0],
        dtype, hidden_size, batch_size)
    x.reverse()
    state_b_h_tensors, state_b_c_tensor = _generate_one_direction_LSTM(
        transformer, x, w[1], r[1], b[1], initial_h[1], initial_c[1], p[1], clip, act[1],
        dtype, hidden_size, batch_size)
    state_b_h_tensors.reverse()

    y_direction_dim = layout + 1
    y_c_direction_dim = layout
    state_layout_tensors = []
    seq_length_dim = layout
    for f_h_state, b_h_state in zip(state_f_h_tensors, state_b_h_tensors):
        state_f_layout_tensors = transformer.make_unsqueeze(
            f_h_state, axes=[seq_length_dim, y_direction_dim])
        state_b_layout_tensors = transformer.make_unsqueeze(
            b_h_state, axes=[seq_length_dim, y_direction_dim])
        state_layout_tensors += [
            transformer.make_concat([state_f_layout_tensors, state_b_layout_tensors],
                                    axis=y_direction_dim)
        ]

    last_f_state_layout_tensor = transformer.make_unsqueeze(state_f_h_tensors[-1],
                                                            axes=[y_c_direction_dim])
    last_b_state_layout_tensor = transformer.make_unsqueeze(state_b_h_tensors[0],
                                                            axes=[y_c_direction_dim])

    Y_h = outputs[1]
    transformer.make_node('Concat',
                          [last_f_state_layout_tensor, last_b_state_layout_tensor], [Y_h],
                          axis=y_c_direction_dim)

    Y_f_c = transformer.make_unsqueeze(state_f_c_tensor, axes=[y_c_direction_dim])
    Y_b_c = transformer.make_unsqueeze(state_b_c_tensor, axes=[y_c_direction_dim])
    Y_c = outputs[2]
    transformer.make_node('Concat', [Y_f_c, Y_b_c], [Y_c], axis=y_c_direction_dim)

    Y = outputs[0]
    transformer.make_node('Concat', state_layout_tensors, [Y], axis=seq_length_dim)


def _legalize_LSTM(transformer, tensor_infos, node):
    """Unroll LSTM operation

    Args:
        transformer (_ModelTransformerHelper): transformation helper
        tensor_infos (dict from str to _TensorInfo): dict maps tensor name to it's shape and dtype info
        node (onnx.onnx_ml_pb2.NodeProto): LSTM operation to unroll
    """
    inputs = node.input
    if len(inputs) > 4 and inputs[4] != '':
        raise NotImplementedError('Variadic length of output is not supported')
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

    if len(activation_alpha) > 0 or len(activation_beta) > 0:
        raise NotImplementedError('Unsupported parameters for LSTM activations')

    for act in activations:
        if act not in ['Relu', 'Tanh', 'Sigmoid']:
            raise NotImplementedError('Unsupported activation function')

    if input_forget != 0:
        raise NotImplementedError('Unsupported input_forget attribute value')

    seq_length_dim = layout
    seq_length = tensor_infos[inputs[0]].shape[seq_length_dim]
    if hidden_size == 0:
        hidden_size = tensor_infos[inputs[2]].shape[2]

    input_split_tensor = transformer.make_split(inputs[0],
                                                split_sizes=[1] * seq_length,
                                                axis=seq_length_dim)
    x = []
    for i in range(len(input_split_tensor)):
        input_frame_tensor = input_split_tensor[i]
        squeezed_frame_tensor = transformer.make_squeeze(input_frame_tensor, axes=[0])
        x += [squeezed_frame_tensor]

    if direction in ['forward', 'reverse']:
        _transform_unidirectional_LSTM(transformer, node, x, tensor_infos, activations,
                                       clip, direction, hidden_size, layout)
    elif direction == 'bidirectional':
        _transform_bidirectional_LSTM(transformer, node, x, tensor_infos, activations,
                                      clip, hidden_size, layout)
    else:
        raise RuntimeError('Unknown LSTM type')

    transformer.mark_for_deletion(node)


def legalize(model, options):
    """Replace selected operations in onnx model

    Replaces operations, selected by given options with different operation sequences.
    For example remove unsupported parts of graph with sequences of supported operations.

    Note that graph is changes inplace.

    Args:
        model (onnx.onnx_ml_pb2.ModelProto): target model
        options (LegalizeOptions):
    """
    tensor_infos = _get_tensor_infos(model)

    transformer = _ModelTransformerHelper(model)

    node_id = 0
    while node_id < len(model.graph.node):
        node = model.graph.node[node_id]
        if node.op_type == 'RNN' and options.unroll_rnn:
            # opset version is required by split operation
            if model.opset_import[0].version >= 13:
                raise NotImplementedError(
                    'Can not generate code with opcode version 13 and greater')
            transformer.set_insert_id(node_id)
            _legalize_RNN(transformer, tensor_infos, node)
            node_id = transformer.get_insert_id()
        elif node.op_type == 'LSTM' and options.unroll_lstm:
            if model.opset_import[0].version >= 13:
                raise NotImplementedError(
                    'Can not generate code with opcode version 13 and greater')
            transformer.set_insert_id(node_id)
            _legalize_LSTM(transformer, tensor_infos, node)
            node_id = transformer.get_insert_id()
        node_id += 1

    transformer.delete_marked_nodes()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            'usage: ./legalize_onnx.py <path to input model> <path to output model>\n'
            '\n'
            '    In stand-alone utility mode this tool provides basic funtionality\n'
            '    If you want to have more control over applied transformations, use this legalizer as a library'
        )
        exit(1)
    options = LegalizeOptions()
    options.unroll_lstm = True
    options.unroll_rnn = True
    model = onnx.load(sys.argv[1])
    legalize(model, options)
    onnx.save(model, sys.argv[2])
