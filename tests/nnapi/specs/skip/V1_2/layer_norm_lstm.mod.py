#
# Copyright (C) 2018 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy

# LSTM Test: Layer Normalization, No Cifg, Peephole, Projection, and No Clipping.
model = Model()

n_batch = 2
n_input = 5
n_cell = 4
n_output = 3

input = Input("input", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_input))

input_to_input_weights = Input("input_to_input_weights", "TENSOR_FLOAT32",
                               "{%d, %d}" % (n_cell, n_input))
input_to_forget_weights = Input("input_to_forget_weights", "TENSOR_FLOAT32",
                                "{%d, %d}" % (n_cell, n_input))
input_to_cell_weights = Input("input_to_cell_weights", "TENSOR_FLOAT32",
                              "{%d, %d}" % (n_cell, n_input))
input_to_output_weights = Input("input_to_output_weights", "TENSOR_FLOAT32",
                                "{%d, %d}" % (n_cell, n_input))

recurrent_to_input_weights = Input("recurrent_to_intput_weights",
                                   "TENSOR_FLOAT32",
                                   "{%d, %d}" % (n_cell, n_output))
recurrent_to_forget_weights = Input("recurrent_to_forget_weights",
                                    "TENSOR_FLOAT32",
                                    "{%d, %d}" % (n_cell, n_output))
recurrent_to_cell_weights = Input("recurrent_to_cell_weights", "TENSOR_FLOAT32",
                                  "{%d, %d}" % (n_cell, n_output))
recurrent_to_output_weights = Input("recurrent_to_output_weights",
                                    "TENSOR_FLOAT32",
                                    "{%d, %d}" % (n_cell, n_output))

cell_to_input_weights = Input("cell_to_input_weights", "TENSOR_FLOAT32",
                              "{%d}" % (n_cell))
cell_to_forget_weights = Input("cell_to_forget_weights", "TENSOR_FLOAT32",
                               "{%d}" % (n_cell))
cell_to_output_weights = Input("cell_to_output_weights", "TENSOR_FLOAT32",
                               "{%d}" % (n_cell))

input_gate_bias = Input("input_gate_bias", "TENSOR_FLOAT32", "{%d}" % (n_cell))
forget_gate_bias = Input("forget_gate_bias", "TENSOR_FLOAT32",
                         "{%d}" % (n_cell))
cell_gate_bias = Input("cell_gate_bias", "TENSOR_FLOAT32", "{%d}" % (n_cell))
output_gate_bias = Input("output_gate_bias", "TENSOR_FLOAT32",
                         "{%d}" % (n_cell))

projection_weights = Input("projection_weights", "TENSOR_FLOAT32",
                           "{%d,%d}" % (n_output, n_cell))
projection_bias = Input("projection_bias", "TENSOR_FLOAT32", "{0}")

output_state_in = Input("output_state_in", "TENSOR_FLOAT32",
                        "{%d, %d}" % (n_batch, n_output))
cell_state_in = Input("cell_state_in", "TENSOR_FLOAT32",
                      "{%d, %d}" % (n_batch, n_cell))

activation_param = Int32Scalar("activation_param", 4)  # Tanh
cell_clip_param = Float32Scalar("cell_clip_param", 0.)
proj_clip_param = Float32Scalar("proj_clip_param", 0.)

input_layer_norm_weights = Input("input_layer_norm_weights", "TENSOR_FLOAT32",
                                 "{%d}" % n_cell)
forget_layer_norm_weights = Input("forget_layer_norm_weights", "TENSOR_FLOAT32",
                                  "{%d}" % n_cell)
cell_layer_norm_weights = Input("cell_layer_norm_weights", "TENSOR_FLOAT32",
                                "{%d}" % n_cell)
output_layer_norm_weights = Input("output_layer_norm_weights", "TENSOR_FLOAT32",
                                  "{%d}" % n_cell)

scratch_buffer = IgnoredOutput("scratch_buffer", "TENSOR_FLOAT32",
                               "{%d, %d}" % (n_batch, (n_cell * 4)))
output_state_out = Output("output_state_out", "TENSOR_FLOAT32",
                          "{%d, %d}" % (n_batch, n_output))
cell_state_out = Output("cell_state_out", "TENSOR_FLOAT32",
                        "{%d, %d}" % (n_batch, n_cell))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_output))

model = model.Operation(
    "LSTM", input, input_to_input_weights, input_to_forget_weights,
    input_to_cell_weights, input_to_output_weights, recurrent_to_input_weights,
    recurrent_to_forget_weights, recurrent_to_cell_weights,
    recurrent_to_output_weights, cell_to_input_weights, cell_to_forget_weights,
    cell_to_output_weights, input_gate_bias, forget_gate_bias, cell_gate_bias,
    output_gate_bias, projection_weights, projection_bias, output_state_in,
    cell_state_in, activation_param, cell_clip_param, proj_clip_param,
    input_layer_norm_weights, forget_layer_norm_weights,
    cell_layer_norm_weights, output_layer_norm_weights).To(
        [scratch_buffer, output_state_out, cell_state_out, output])

# Example 1. Input in operand 0,
input0 = {
    input_to_input_weights: [
        0.5, 0.6, 0.7, -0.8, -0.9, 0.1, 0.2, 0.3, -0.4, 0.5, -0.8, 0.7, -0.6,
        0.5, -0.4, -0.5, -0.4, -0.3, -0.2, -0.1
    ],
    input_to_forget_weights: [
        -0.6, -0.1, 0.3, 0.2, 0.9, -0.5, -0.2, -0.4, 0.3, -0.8, -0.4, 0.3, -0.5,
        -0.4, -0.6, 0.3, -0.4, -0.6, -0.5, -0.5
    ],
    input_to_cell_weights: [
        -0.4, -0.3, -0.2, -0.1, -0.5, 0.5, -0.2, -0.3, -0.2, -0.6, 0.6, -0.1,
        -0.4, -0.3, -0.7, 0.7, -0.9, -0.5, 0.8, 0.6
    ],
    input_to_output_weights: [
        -0.8, -0.4, -0.2, -0.9, -0.1, -0.7, 0.3, -0.3, -0.8, -0.2, 0.6, -0.2,
        0.4, -0.7, -0.3, -0.5, 0.1, 0.5, -0.6, -0.4
    ],
    input_gate_bias: [0.03, 0.15, 0.22, 0.38],
    forget_gate_bias: [0.1, -0.3, -0.2, 0.1],
    cell_gate_bias: [-0.05, 0.72, 0.25, 0.08],
    output_gate_bias: [0.05, -0.01, 0.2, 0.1],
    recurrent_to_input_weights: [
        -0.2, -0.3, 0.4, 0.1, -0.5, 0.9, -0.2, -0.3, -0.7, 0.05, -0.2, -0.6
    ],
    recurrent_to_cell_weights: [
        -0.3, 0.2, 0.1, -0.3, 0.8, -0.08, -0.2, 0.3, 0.8, -0.6, -0.1, 0.2
    ],
    recurrent_to_forget_weights: [
        -0.5, -0.3, -0.5, -0.2, 0.6, 0.4, 0.9, 0.3, -0.1, 0.2, 0.5, 0.2
    ],
    recurrent_to_output_weights: [
        0.3, -0.1, 0.1, -0.2, -0.5, -0.7, -0.2, -0.6, -0.1, -0.4, -0.7, -0.2
    ],
    cell_to_input_weights: [0.05, 0.1, 0.25, 0.15],
    cell_to_forget_weights: [-0.02, -0.15, -0.25, -0.03],
    cell_to_output_weights: [0.1, -0.1, -0.5, 0.05],
    projection_weights: [
        -0.1, 0.2, 0.01, -0.2, 0.1, 0.5, 0.3, 0.08, 0.07, 0.2, -0.4, 0.2
    ],
    projection_bias: [],
    input_layer_norm_weights: [0.1, 0.2, 0.3, 0.5],
    forget_layer_norm_weights: [0.2, 0.2, 0.4, 0.3],
    cell_layer_norm_weights: [0.7, 0.2, 0.3, 0.8],
    output_layer_norm_weights: [0.6, 0.2, 0.2, 0.5]
}

test_inputs = [[0.7, 0.8, 0.1, 0.2, 0.3, 0.3, 0.2, 0.9, 0.8, 0.1],
               [0.8, 0.1, 0.2, 0.4, 0.5, 0.1, 0.5, 0.2, 0.4, 0.2],
               [0.2, 0.7, 0.7, 0.1, 0.7, 0.6, 0.9, 0.2, 0.5, 0.7]]
golden_cell_states = [
    [
        -0.451771229505539, 0.376915663480759, 0.225425109267235, 0.232406347990036, -0.252585828304291, 0.330421179533005, 0.017305245622993, 0.366601228713989
    ],
    [
        -0.645632147789001, 0.518238246440887, 0.168679088354111, 0.555787742137909, -0.493674814701080, 0.475847363471985, 0.106874041259289, 0.504309654235840
    ],
    [-0.742560744285583, 0.579139292240143, 0.114988230168819, 0.649957716464996, -0.686565399169922, 0.548869132995605, 0.173138767480850, 0.587379336357117],
]
cell_states = [[0, 0, 0, 0, 0, 0, 0, 0]] + golden_cell_states[:2]

golden_outputs = [
    [0.024407668039203, 0.128027379512787, -0.001709178090096, -0.006924282759428, 0.084874063730240, 0.063444979488850],
    [0.013764165341854, 0.140751048922539, 0.039583537727594, -0.004039138555527, 0.139963015913963, 0.072681039571762],
    [-0.004592306911945, 0.155278354883194, 0.083737745881081, 0.007527053356171, 0.161902531981468, 0.056137066334486],
]
output_states = [[0, 0, 0, 0, 0, 0]] + golden_outputs[:2]

tests = zip(
    test_inputs, output_states, cell_states, golden_cell_states, golden_outputs)

for test_input, output_state, cell_state, golden_state, golden_output in tests:
  cur_input = copy.deepcopy(input0)
  cur_input[input] = test_input
  cur_input[output_state_in] = output_state
  cur_input[cell_state_in] = cell_state
  cur_output = {
      scratch_buffer: [0] * (n_batch * n_cell * 4),
      cell_state_out: golden_state,
      output_state_out: golden_output,
      output: golden_output
  }
  Example((cur_input, cur_output), name="NoCifgPeepholeProjectionNoClippingLayerNormLstm")


# LSTM Test: Layer Normalization, Cifg, Peephole, Projection, and No Clipping.
model = Model()

n_batch = 2
n_input = 5
n_cell = 4
n_output = 3

input = Input("input", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_input))

input_to_input_weights = Input("input_to_input_weights", "TENSOR_FLOAT32",
                               "{0, 0}")
input_to_forget_weights = Input("input_to_forget_weights", "TENSOR_FLOAT32",
                                "{%d, %d}" % (n_cell, n_input))
input_to_cell_weights = Input("input_to_cell_weights", "TENSOR_FLOAT32",
                              "{%d, %d}" % (n_cell, n_input))
input_to_output_weights = Input("input_to_output_weights", "TENSOR_FLOAT32",
                                "{%d, %d}" % (n_cell, n_input))

recurrent_to_input_weights = Input("recurrent_to_intput_weights",
                                   "TENSOR_FLOAT32",
                                   "{0, 0}")
recurrent_to_forget_weights = Input("recurrent_to_forget_weights",
                                    "TENSOR_FLOAT32",
                                    "{%d, %d}" % (n_cell, n_output))
recurrent_to_cell_weights = Input("recurrent_to_cell_weights", "TENSOR_FLOAT32",
                                  "{%d, %d}" % (n_cell, n_output))
recurrent_to_output_weights = Input("recurrent_to_output_weights",
                                    "TENSOR_FLOAT32",
                                    "{%d, %d}" % (n_cell, n_output))

cell_to_input_weights = Input("cell_to_input_weights", "TENSOR_FLOAT32",
                              "{0}")
cell_to_forget_weights = Input("cell_to_forget_weights", "TENSOR_FLOAT32",
                               "{%d}" % (n_cell))
cell_to_output_weights = Input("cell_to_output_weights", "TENSOR_FLOAT32",
                               "{%d}" % (n_cell))

input_gate_bias = Input("input_gate_bias", "TENSOR_FLOAT32", "{0}")
forget_gate_bias = Input("forget_gate_bias", "TENSOR_FLOAT32",
                         "{%d}" % (n_cell))
cell_gate_bias = Input("cell_gate_bias", "TENSOR_FLOAT32", "{%d}" % (n_cell))
output_gate_bias = Input("output_gate_bias", "TENSOR_FLOAT32",
                         "{%d}" % (n_cell))

projection_weights = Input("projection_weights", "TENSOR_FLOAT32",
                           "{%d,%d}" % (n_output, n_cell))
projection_bias = Input("projection_bias", "TENSOR_FLOAT32", "{0}")

output_state_in = Input("output_state_in", "TENSOR_FLOAT32",
                        "{%d, %d}" % (n_batch, n_output))
cell_state_in = Input("cell_state_in", "TENSOR_FLOAT32",
                      "{%d, %d}" % (n_batch, n_cell))

activation_param = Int32Scalar("activation_param", 4)  # Tanh
cell_clip_param = Float32Scalar("cell_clip_param", 0.)
proj_clip_param = Float32Scalar("proj_clip_param", 0.)

input_layer_norm_weights = Input("input_layer_norm_weights", "TENSOR_FLOAT32",
                                 "{0}")
forget_layer_norm_weights = Input("forget_layer_norm_weights", "TENSOR_FLOAT32",
                                  "{%d}" % n_cell)
cell_layer_norm_weights = Input("cell_layer_norm_weights", "TENSOR_FLOAT32",
                                "{%d}" % n_cell)
output_layer_norm_weights = Input("output_layer_norm_weights", "TENSOR_FLOAT32",
                                  "{%d}" % n_cell)

scratch_buffer = IgnoredOutput("scratch_buffer", "TENSOR_FLOAT32",
                               "{%d, %d}" % (n_batch, (n_cell * 3)))
output_state_out = IgnoredOutput("output_state_out", "TENSOR_FLOAT32",
                         "{%d, %d}" % (n_batch, n_output))
cell_state_out = Output("cell_state_out", "TENSOR_FLOAT32",
                        "{%d, %d}" % (n_batch, n_cell))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_output))

model = model.Operation(
    "LSTM", input, input_to_input_weights, input_to_forget_weights,
    input_to_cell_weights, input_to_output_weights, recurrent_to_input_weights,
    recurrent_to_forget_weights, recurrent_to_cell_weights,
    recurrent_to_output_weights, cell_to_input_weights, cell_to_forget_weights,
    cell_to_output_weights, input_gate_bias, forget_gate_bias, cell_gate_bias,
    output_gate_bias, projection_weights, projection_bias, output_state_in,
    cell_state_in, activation_param, cell_clip_param, proj_clip_param,
    input_layer_norm_weights, forget_layer_norm_weights,
    cell_layer_norm_weights, output_layer_norm_weights).To(
        [scratch_buffer, output_state_out, cell_state_out, output])

# Example 1. Input in operand 0,
input0 = {
    input_to_input_weights: [],
    input_to_forget_weights: [
        -0.6, -0.1, 0.3, 0.2, 0.9, -0.5, -0.2, -0.4, 0.3, -0.8, -0.4, 0.3, -0.5,
        -0.4, -0.6, 0.3, -0.4, -0.6, -0.5, -0.5
    ],
    input_to_cell_weights: [
        -0.4, -0.3, -0.2, -0.1, -0.5, 0.5, -0.2, -0.3, -0.2, -0.6, 0.6, -0.1,
        -0.4, -0.3, -0.7, 0.7, -0.9, -0.5, 0.8, 0.6
    ],
    input_to_output_weights: [
        -0.8, -0.4, -0.2, -0.9, -0.1, -0.7, 0.3, -0.3, -0.8, -0.2, 0.6, -0.2,
        0.4, -0.7, -0.3, -0.5, 0.1, 0.5, -0.6, -0.4
    ],
    input_gate_bias: [],
    forget_gate_bias: [0.1, -0.3, -0.2, 0.1],
    cell_gate_bias: [-0.05, 0.72, 0.25, 0.08],
    output_gate_bias: [0.05, -0.01, 0.2, 0.1],
    recurrent_to_input_weights: [],
    recurrent_to_cell_weights: [
        -0.3, 0.2, 0.1, -0.3, 0.8, -0.08, -0.2, 0.3, 0.8, -0.6, -0.1, 0.2
    ],
    recurrent_to_forget_weights: [
        -0.5, -0.3, -0.5, -0.2, 0.6, 0.4, 0.9, 0.3, -0.1, 0.2, 0.5, 0.2
    ],
    recurrent_to_output_weights: [
        0.3, -0.1, 0.1, -0.2, -0.5, -0.7, -0.2, -0.6, -0.1, -0.4, -0.7, -0.2
    ],
    cell_to_input_weights: [],
    cell_to_forget_weights: [-0.02, -0.15, -0.25, -0.03],
    cell_to_output_weights: [0.1, -0.1, -0.5, 0.05],
    projection_weights: [
        -0.1, 0.2, 0.01, -0.2, 0.1, 0.5, 0.3, 0.08, 0.07, 0.2, -0.4, 0.2
    ],
    projection_bias: [],
    input_layer_norm_weights: [],
    forget_layer_norm_weights: [0.2, 0.2, 0.4, 0.3],
    cell_layer_norm_weights: [0.7, 0.2, 0.3, 0.8],
    output_layer_norm_weights: [0.6, 0.2, 0.2, 0.5]
}

test_inputs = [[0.7, 0.8, 0.1, 0.2, 0.3, 0.3, 0.2, 0.9, 0.8, 0.1],
               [0.8, 0.1, 0.2, 0.4, 0.5, 0.1, 0.5, 0.2, 0.4, 0.2],
               [0.2, 0.7, 0.7, 0.1, 0.7, 0.6, 0.9, 0.2, 0.5, 0.7]]
golden_cell_states = [
    [
        -0.3510298, 0.4261035, 0.2146365, 0.2771652, -0.1885517, 0.3252200, 0.0203665, 0.4896766
    ],
    [
        -0.5069088, 0.5386363, 0.1980069, 0.5355753, -0.3866257, 0.4749442, 0.1074765, 0.7124508
    ],
    [
        -0.5736622, 0.5952501, 0.1292950, 0.7110270, -0.5323033, 0.5556133, 0.1800992, 0.7845056
    ],
]
cell_states = [[0, 0, 0, 0, 0, 0, 0, 0]] + golden_cell_states[:2]

golden_outputs = [
    [0.02129706, 0.140816242, 0.0112733059, -0.0226350538, 0.0916948169, 0.0769175813],
    [0.0132302344, 0.152308047, 0.0346313119, -0.0269966982, 0.149707705, 0.094149217],
    [-0.0123688057, 0.165790111, 0.0893077999, -0.0103429332, 0.173016444, 0.0720508844],
]
output_states = [[0, 0, 0, 0, 0, 0]] + golden_outputs[:2]

tests = zip(
    test_inputs, output_states, cell_states, golden_cell_states, golden_outputs)

for test_input, output_state, cell_state, golden_state, golden_output in tests:
  cur_input = copy.deepcopy(input0)
  cur_input[input] = test_input
  cur_input[output_state_in] = output_state
  cur_input[cell_state_in] = cell_state
  cur_output = {
      scratch_buffer: [0] * (n_batch * n_cell * 3),
      cell_state_out: golden_state,
      output_state_out: golden_output,
      output: golden_output
  }
  Example((cur_input, cur_output), name="CifgPeepholeProjectionNoClippingLayerNormLstm")
