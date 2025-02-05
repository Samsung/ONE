#
# Copyright (C) 2019 The Android Open Source Project
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

# Bidirectional Sequence LSTM Test:
# FLOAT32, Layer Normalization, No Cifg, Peephole, Projection, and No Clipping.
# Verifies forward output only.

n_batch = 2
n_input = 5
n_cell = 4
n_output = 3
max_time = 3

input = Input("input", "TENSOR_FLOAT32", "{{{}, {}, {}}}".format(max_time, n_batch, n_input))

fw_input_to_input_weights = Input(
    "fw_input_to_input_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
fw_input_to_forget_weights = Input(
    "fw_input_to_forget_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
fw_input_to_cell_weights = Input(
    "fw_input_to_cell_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
fw_input_to_output_weights = Input(
    "fw_input_to_output_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))

fw_recurrent_to_input_weights = Input(
    "fw_recurrent_to_input_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))
fw_recurrent_to_forget_weights = Input(
    "fw_recurrent_to_forget_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))
fw_recurrent_to_cell_weights = Input(
    "fw_recurrent_to_cell_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))
fw_recurrent_to_output_weights = Input(
    "fw_recurrent_to_output_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))

fw_cell_to_input_weights = Input(
    "fw_cell_to_input_weights", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
fw_cell_to_forget_weights = Input(
    "fw_cell_to_forget_weights", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
fw_cell_to_output_weights = Input(
    "fw_cell_to_output_weights", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))

fw_input_gate_bias = Input(
    "fw_input_gate_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
fw_forget_gate_bias = Input(
    "fw_forget_gate_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
fw_cell_bias = Input(
    "fw_cell_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
fw_output_gate_bias = Input(
    "fw_output_gate_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))

fw_projection_weights = Input(
    "fw_projection_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_output, n_cell))
fw_projection_bias = Input(
    "fw_projection_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_output))

bw_input_to_input_weights = Input(
    "bw_input_to_input_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
bw_input_to_forget_weights = Input(
    "bw_input_to_forget_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
bw_input_to_cell_weights = Input(
    "bw_input_to_cell_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
bw_input_to_output_weights = Input(
    "bw_input_to_output_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))

bw_recurrent_to_input_weights = Input(
    "bw_recurrent_to_input_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))
bw_recurrent_to_forget_weights = Input(
    "bw_recurrent_to_forget_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))
bw_recurrent_to_cell_weights = Input(
    "bw_recurrent_to_cell_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))
bw_recurrent_to_output_weights = Input(
    "bw_recurrent_to_output_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_output))

bw_cell_to_input_weights = Input(
    "bw_cell_to_input_weights", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
bw_cell_to_forget_weights = Input(
    "bw_cell_to_forget_weights", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
bw_cell_to_output_weights = Input(
    "bw_cell_to_output_weights", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))

bw_input_gate_bias = Input(
    "bw_input_gate_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
bw_forget_gate_bias = Input(
    "bw_forget_gate_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
bw_cell_bias = Input(
    "bw_cell_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))
bw_output_gate_bias = Input(
    "bw_output_gate_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_cell))

bw_projection_weights = Input(
    "bw_projection_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_output, n_cell))
bw_projection_bias = Input(
    "bw_projection_bias", "TENSOR_FLOAT32", "{{{}}}".format(n_output))

fw_activation_state = Input(
    "fw_activatiom_state", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_batch, n_output))
fw_cell_state = Input(
    "fw_cell_state", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_batch, n_cell))

bw_activation_state = Input(
    "bw_activatiom_state", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_batch, n_output))
bw_cell_state = Input(
    "bw_cell_state", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_batch, n_cell))

aux_input = Input("input", "TENSOR_FLOAT32", "{{{}, {}, {}}}".format(max_time, n_batch, n_input))

fw_aux_input_to_input_weights = Input(
    "fw_aux_input_to_input_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
fw_aux_input_to_forget_weights = Input(
    "fw_input_to_forget_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
fw_aux_input_to_cell_weights = Input(
    "fw_aux_input_to_cell_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
fw_aux_input_to_output_weights = Input(
    "fw_aux_input_to_output_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))

bw_aux_input_to_input_weights = Input(
    "bw_aux_input_to_input_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
bw_aux_input_to_forget_weights = Input(
    "bw_input_to_forget_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
bw_aux_input_to_cell_weights = Input(
    "bw_aux_input_to_cell_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))
bw_aux_input_to_output_weights = Input(
    "bw_aux_input_to_output_weights", "TENSOR_FLOAT32", "{{{}, {}}}".format(n_cell, n_input))

fw_input_layer_norm_weights = Input("input_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)
fw_forget_layer_norm_weights = Input("forget_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)
fw_cell_layer_norm_weights = Input("cell_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)
fw_output_layer_norm_weights = Input("output_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)

bw_input_layer_norm_weights = Input("input_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)
bw_forget_layer_norm_weights = Input("forget_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)
bw_cell_layer_norm_weights = Input("cell_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)
bw_output_layer_norm_weights = Input("output_layer_norm_weights", "TENSOR_FLOAT32", "{%d}" % n_cell)

fw_output=Output("fw_output", "TENSOR_FLOAT32", "{{{}, {}, {}}}".format(max_time, n_batch, n_output))
bw_output=IgnoredOutput("bw_output", "TENSOR_FLOAT32", "{{{}, {}, {}}}".format(max_time, n_batch, n_output))

def test(
    name,
    input_data=[],
    fw_input_to_input_weights_data=[],
    fw_input_to_forget_weights_data=[],
    fw_input_to_cell_weights_data=[],
    fw_input_to_output_weights_data=[],
    fw_recurrent_to_input_weights_data=[],
    fw_recurrent_to_forget_weights_data=[],
    fw_recurrent_to_cell_weights_data=[],
    fw_recurrent_to_output_weights_data=[],
    fw_cell_to_input_weights_data=[],
    fw_cell_to_forget_weights_data=[],
    fw_cell_to_output_weights_data=[],
    fw_input_gate_bias_data=[],
    fw_forget_gate_bias_data=[],
    fw_cell_bias_data=[],
    fw_output_gate_bias_data=[],
    fw_projection_weights_data=[],
    fw_projection_bias_data=[],
    bw_input_to_input_weights_data=[],
    bw_input_to_forget_weights_data=[],
    bw_input_to_cell_weights_data=[],
    bw_input_to_output_weights_data=[],
    bw_recurrent_to_input_weights_data=[],
    bw_recurrent_to_forget_weights_data=[],
    bw_recurrent_to_cell_weights_data=[],
    bw_recurrent_to_output_weights_data=[],
    bw_cell_to_input_weights_data=[],
    bw_cell_to_forget_weights_data=[],
    bw_cell_to_output_weights_data=[],
    bw_input_gate_bias_data=[],
    bw_forget_gate_bias_data=[],
    bw_cell_bias_data=[],
    bw_output_gate_bias_data=[],
    bw_projection_weights_data=[],
    bw_projection_bias_data=[],
    fw_activation_state_data=[],
    fw_cell_state_data=[],
    bw_activation_state_data=[],
    bw_cell_state_data=[],
    aux_input_data=[],
    fw_aux_input_to_input_weights_data=[],
    fw_aux_input_to_forget_weights_data=[],
    fw_aux_input_to_cell_weights_data=[],
    fw_aux_input_to_output_weights_data=[],
    bw_aux_input_to_input_weights_data=[],
    bw_aux_input_to_forget_weights_data=[],
    bw_aux_input_to_cell_weights_data=[],
    bw_aux_input_to_output_weights_data=[],
    fw_input_layer_norm_weights_data=[],
    fw_forget_layer_norm_weights_data=[],
    fw_cell_layer_norm_weights_data=[],
    fw_output_layer_norm_weights_data=[],
    bw_input_layer_norm_weights_data=[],
    bw_forget_layer_norm_weights_data=[],
    bw_cell_layer_norm_weights_data=[],
    bw_output_layer_norm_weights_data=[],
    fw_output_data=[],
    bw_output_data=[],):

  activation = Int32Scalar("activation", 4)
  cell_clip = Float32Scalar("cell_clip", 0.0)
  proj_clip = Float32Scalar("proj_clip", 0.0)
  merge_outputs = BoolScalar("merge_outputs", False)
  time_major = BoolScalar("time_major", True)

  model = Model().Operation(
      "BIDIRECTIONAL_SEQUENCE_LSTM",
      input,
      fw_input_to_input_weights,
      fw_input_to_forget_weights,
      fw_input_to_cell_weights,
      fw_input_to_output_weights,
      fw_recurrent_to_input_weights,
      fw_recurrent_to_forget_weights,
      fw_recurrent_to_cell_weights,
      fw_recurrent_to_output_weights,
      fw_cell_to_input_weights,
      fw_cell_to_forget_weights,
      fw_cell_to_output_weights,
      fw_input_gate_bias,
      fw_forget_gate_bias,
      fw_cell_bias,
      fw_output_gate_bias,
      fw_projection_weights,
      fw_projection_bias,
      bw_input_to_input_weights,
      bw_input_to_forget_weights,
      bw_input_to_cell_weights,
      bw_input_to_output_weights,
      bw_recurrent_to_input_weights,
      bw_recurrent_to_forget_weights,
      bw_recurrent_to_cell_weights,
      bw_recurrent_to_output_weights,
      bw_cell_to_input_weights,
      bw_cell_to_forget_weights,
      bw_cell_to_output_weights,
      bw_input_gate_bias,
      bw_forget_gate_bias,
      bw_cell_bias,
      bw_output_gate_bias,
      bw_projection_weights,
      bw_projection_bias,
      fw_activation_state,
      fw_cell_state,
      bw_activation_state,
      bw_cell_state,
      aux_input,
      fw_aux_input_to_input_weights,
      fw_aux_input_to_forget_weights,
      fw_aux_input_to_cell_weights,
      fw_aux_input_to_output_weights,
      bw_aux_input_to_input_weights,
      bw_aux_input_to_forget_weights,
      bw_aux_input_to_cell_weights,
      bw_aux_input_to_output_weights,
      activation, cell_clip, proj_clip, merge_outputs, time_major,
      fw_input_layer_norm_weights,
      fw_forget_layer_norm_weights,
      fw_cell_layer_norm_weights,
      fw_output_layer_norm_weights,
      bw_input_layer_norm_weights,
      bw_forget_layer_norm_weights,
      bw_cell_layer_norm_weights,
      bw_output_layer_norm_weights,).To(fw_output, bw_output)

  example = Example(
      {
          input: input_data,
          fw_input_to_input_weights: fw_input_to_input_weights_data,
          fw_input_to_forget_weights: fw_input_to_forget_weights_data,
          fw_input_to_cell_weights: fw_input_to_cell_weights_data,
          fw_input_to_output_weights: fw_input_to_output_weights_data,
          fw_recurrent_to_input_weights: fw_recurrent_to_input_weights_data,
          fw_recurrent_to_forget_weights: fw_recurrent_to_forget_weights_data,
          fw_recurrent_to_cell_weights: fw_recurrent_to_cell_weights_data,
          fw_recurrent_to_output_weights: fw_recurrent_to_output_weights_data,
          fw_cell_to_input_weights: fw_cell_to_input_weights_data,
          fw_cell_to_forget_weights: fw_cell_to_forget_weights_data,
          fw_cell_to_output_weights: fw_cell_to_output_weights_data,
          fw_input_gate_bias: fw_input_gate_bias_data,
          fw_forget_gate_bias: fw_forget_gate_bias_data,
          fw_cell_bias: fw_cell_bias_data,
          fw_output_gate_bias: fw_output_gate_bias_data,
          fw_projection_weights: fw_projection_weights_data,
          fw_projection_bias: fw_projection_bias_data,
          bw_input_to_input_weights: bw_input_to_input_weights_data,
          bw_input_to_forget_weights: bw_input_to_forget_weights_data,
          bw_input_to_cell_weights: bw_input_to_cell_weights_data,
          bw_input_to_output_weights: bw_input_to_output_weights_data,
          bw_recurrent_to_input_weights: bw_recurrent_to_input_weights_data,
          bw_recurrent_to_forget_weights: bw_recurrent_to_forget_weights_data,
          bw_recurrent_to_cell_weights: bw_recurrent_to_cell_weights_data,
          bw_recurrent_to_output_weights: bw_recurrent_to_output_weights_data,
          bw_cell_to_input_weights: bw_cell_to_input_weights_data,
          bw_cell_to_forget_weights: bw_cell_to_forget_weights_data,
          bw_cell_to_output_weights: bw_cell_to_output_weights_data,
          bw_input_gate_bias: bw_input_gate_bias_data,
          bw_forget_gate_bias: bw_forget_gate_bias_data,
          bw_cell_bias: bw_cell_bias_data,
          bw_output_gate_bias: bw_output_gate_bias_data,
          bw_projection_weights: bw_projection_weights_data,
          bw_projection_bias: bw_projection_bias_data,
          fw_activation_state: fw_activation_state_data,
          fw_cell_state: fw_cell_state_data,
          bw_activation_state: bw_activation_state_data,
          bw_cell_state: bw_cell_state_data,
          aux_input: aux_input_data,
          fw_aux_input_to_input_weights: fw_aux_input_to_input_weights_data,
          fw_aux_input_to_forget_weights: fw_aux_input_to_forget_weights_data,
          fw_aux_input_to_cell_weights: fw_aux_input_to_cell_weights_data,
          fw_aux_input_to_output_weights: fw_aux_input_to_output_weights_data,
          bw_aux_input_to_input_weights: bw_aux_input_to_input_weights_data,
          bw_aux_input_to_forget_weights: bw_aux_input_to_forget_weights_data,
          bw_aux_input_to_cell_weights: bw_aux_input_to_cell_weights_data,
          bw_aux_input_to_output_weights: bw_aux_input_to_output_weights_data,
          fw_input_layer_norm_weights: fw_input_layer_norm_weights_data,
          fw_forget_layer_norm_weights: fw_forget_layer_norm_weights_data,
          fw_cell_layer_norm_weights: fw_cell_layer_norm_weights_data,
          fw_output_layer_norm_weights: fw_output_layer_norm_weights_data,
          bw_input_layer_norm_weights: bw_input_layer_norm_weights_data,
          bw_forget_layer_norm_weights: bw_forget_layer_norm_weights_data,
          bw_cell_layer_norm_weights: bw_cell_layer_norm_weights_data,
          bw_output_layer_norm_weights: bw_output_layer_norm_weights_data,
          fw_output: fw_output_data,
          bw_output: bw_output_data,
      },
      model=model, name=name)


fw_input_to_input_weights_data = [
    0.5, 0.6, 0.7, -0.8, -0.9, 0.1, 0.2, 0.3, -0.4, 0.5, -0.8, 0.7, -0.6,
    0.5, -0.4, -0.5, -0.4, -0.3, -0.2, -0.1
]
bw_input_to_input_weights_data = fw_input_to_input_weights_data

fw_input_to_forget_weights_data = [
    -0.6, -0.1, 0.3, 0.2, 0.9, -0.5, -0.2, -0.4, 0.3, -0.8, -0.4, 0.3, -0.5,
    -0.4, -0.6, 0.3, -0.4, -0.6, -0.5, -0.5
]
bw_input_to_forget_weights_data = fw_input_to_forget_weights_data

fw_input_to_cell_weights_data = [
    -0.4, -0.3, -0.2, -0.1, -0.5, 0.5, -0.2, -0.3, -0.2, -0.6, 0.6, -0.1,
    -0.4, -0.3, -0.7, 0.7, -0.9, -0.5, 0.8, 0.6
]
bw_input_to_cell_weights_data = fw_input_to_cell_weights_data

fw_input_to_output_weights_data = [
    -0.8, -0.4, -0.2, -0.9, -0.1, -0.7, 0.3, -0.3, -0.8, -0.2, 0.6, -0.2,
    0.4, -0.7, -0.3, -0.5, 0.1, 0.5, -0.6, -0.4
]
bw_input_to_output_weights_data = fw_input_to_output_weights_data

fw_recurrent_to_input_weights_data = [
    -0.2, -0.3, 0.4, 0.1, -0.5, 0.9, -0.2, -0.3, -0.7, 0.05, -0.2, -0.6
]
bw_recurrent_to_input_weights_data = fw_recurrent_to_input_weights_data

fw_recurrent_to_forget_weights_data = [
    -0.5, -0.3, -0.5, -0.2, 0.6, 0.4, 0.9, 0.3, -0.1, 0.2, 0.5, 0.2
]
bw_recurrent_to_forget_weights_data = fw_recurrent_to_forget_weights_data

fw_recurrent_to_cell_weights_data = [
    -0.3, 0.2, 0.1, -0.3, 0.8, -0.08, -0.2, 0.3, 0.8, -0.6, -0.1, 0.2
]
bw_recurrent_to_cell_weights_data = fw_recurrent_to_cell_weights_data

fw_recurrent_to_output_weights_data = [
    0.3, -0.1, 0.1, -0.2, -0.5, -0.7, -0.2, -0.6, -0.1, -0.4, -0.7, -0.2
]
bw_recurrent_to_output_weights_data = fw_recurrent_to_output_weights_data

fw_cell_to_input_weights_data = [0.05, 0.1, 0.25, 0.15]
bw_cell_to_input_weights_data = fw_cell_to_input_weights_data

fw_cell_to_forget_weights_data = [-0.02, -0.15, -0.25, -0.03]
bw_cell_to_forget_weights_data = fw_cell_to_forget_weights_data

fw_cell_to_output_weights_data = [0.1, -0.1, -0.5, 0.05]
bw_cell_to_output_weights_data = fw_cell_to_output_weights_data

fw_projection_weights_data = [
    -0.1, 0.2, 0.01, -0.2, 0.1, 0.5, 0.3, 0.08, 0.07, 0.2, -0.4, 0.2
]
bw_projection_weights_data = fw_projection_weights_data

fw_input_gate_bias_data = [0.03, 0.15, 0.22, 0.38]
bw_input_gate_bias_data = fw_input_gate_bias_data

fw_forget_gate_bias_data = [0.1, -0.3, -0.2, 0.1]
bw_forget_gate_bias_data = fw_forget_gate_bias_data

fw_cell_bias_data = [-0.05, 0.72, 0.25, 0.08]
bw_cell_bias_data = fw_cell_bias_data

fw_output_gate_bias_data = [0.05, -0.01, 0.2, 0.1]
bw_output_gate_bias_data = fw_output_gate_bias_data

input_layer_norm_weights_data = [0.1, 0.2, 0.3, 0.5]
forget_layer_norm_weights_data = [0.2, 0.2, 0.4, 0.3]
cell_layer_norm_weights_data = [0.7, 0.2, 0.3, 0.8]
output_layer_norm_weights_data = [0.6, 0.2, 0.2, 0.5]

input_data = [0.7, 0.8, 0.1, 0.2, 0.3, 0.3, 0.2, 0.9, 0.8, 0.1,
    0.8, 0.1, 0.2, 0.4, 0.5, 0.1, 0.5, 0.2, 0.4, 0.2,
    0.2, 0.7, 0.7, 0.1, 0.7, 0.6, 0.9, 0.2, 0.5, 0.7]

fw_activation_state_data = [0 for _ in range(n_batch * n_output)]
bw_activation_state_data = [0 for _ in range(n_batch * n_output)]

fw_cell_state_data = [0 for _ in range(n_batch * n_cell)]
bw_cell_state_data = [0 for _ in range(n_batch * n_cell)]

fw_golden_output_data = [
    0.0244077, 0.128027, -0.00170918, -0.00692428, 0.0848741, 0.063445,
    0.0137642, 0.140751, 0.0395835, -0.00403912, 0.139963, 0.072681,
    -0.00459231, 0.155278, 0.0837377, 0.00752706, 0.161903, 0.0561371,
]
bw_golden_output_data = [0 for _ in range(n_batch * max_time * n_output)]

test(
    name="blackbox",
    input_data=input_data,
    fw_input_to_input_weights_data=fw_input_to_input_weights_data,
    fw_input_to_forget_weights_data=fw_input_to_forget_weights_data,
    fw_input_to_cell_weights_data=fw_input_to_cell_weights_data,
    fw_input_to_output_weights_data=fw_input_to_output_weights_data,
    fw_recurrent_to_input_weights_data=fw_recurrent_to_input_weights_data,
    fw_recurrent_to_forget_weights_data=fw_recurrent_to_forget_weights_data,
    fw_recurrent_to_cell_weights_data=fw_recurrent_to_cell_weights_data,
    fw_recurrent_to_output_weights_data=fw_recurrent_to_output_weights_data,
    fw_cell_to_input_weights_data=fw_cell_to_input_weights_data,
    fw_cell_to_forget_weights_data=fw_cell_to_forget_weights_data,
    fw_cell_to_output_weights_data=fw_cell_to_output_weights_data,
    fw_input_gate_bias_data=fw_input_gate_bias_data,
    fw_forget_gate_bias_data=fw_forget_gate_bias_data,
    fw_cell_bias_data=fw_cell_bias_data,
    fw_output_gate_bias_data=fw_output_gate_bias_data,
    fw_projection_weights_data=fw_projection_weights_data,
    bw_input_to_input_weights_data=bw_input_to_input_weights_data,
    bw_input_to_forget_weights_data=bw_input_to_forget_weights_data,
    bw_input_to_cell_weights_data=bw_input_to_cell_weights_data,
    bw_input_to_output_weights_data=bw_input_to_output_weights_data,
    bw_recurrent_to_input_weights_data=bw_recurrent_to_input_weights_data,
    bw_recurrent_to_forget_weights_data=bw_recurrent_to_forget_weights_data,
    bw_recurrent_to_cell_weights_data=bw_recurrent_to_cell_weights_data,
    bw_recurrent_to_output_weights_data=bw_recurrent_to_output_weights_data,
    bw_cell_to_input_weights_data=bw_cell_to_input_weights_data,
    bw_cell_to_forget_weights_data=bw_cell_to_forget_weights_data,
    bw_cell_to_output_weights_data=bw_cell_to_output_weights_data,
    bw_input_gate_bias_data=bw_input_gate_bias_data,
    bw_forget_gate_bias_data=bw_forget_gate_bias_data,
    bw_cell_bias_data=bw_cell_bias_data,
    bw_output_gate_bias_data=bw_output_gate_bias_data,
    bw_projection_weights_data=bw_projection_weights_data,
    fw_activation_state_data = fw_activation_state_data,
    bw_activation_state_data = bw_activation_state_data,
    fw_cell_state_data = fw_cell_state_data,
    bw_cell_state_data = bw_cell_state_data,
    fw_input_layer_norm_weights_data = input_layer_norm_weights_data,
    fw_forget_layer_norm_weights_data = forget_layer_norm_weights_data,
    fw_cell_layer_norm_weights_data = cell_layer_norm_weights_data,
    fw_output_layer_norm_weights_data = output_layer_norm_weights_data,
    bw_input_layer_norm_weights_data = input_layer_norm_weights_data,
    bw_forget_layer_norm_weights_data = forget_layer_norm_weights_data,
    bw_cell_layer_norm_weights_data = cell_layer_norm_weights_data,
    bw_output_layer_norm_weights_data = output_layer_norm_weights_data,
    fw_output_data=fw_golden_output_data,
    bw_output_data=bw_golden_output_data
)
