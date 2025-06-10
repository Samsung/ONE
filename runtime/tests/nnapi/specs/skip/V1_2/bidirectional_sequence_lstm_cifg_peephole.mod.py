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
# FLOAT32, No Layer Normalization, Cifg, Peephole, No Projection, and No Clipping.

n_batch = 1
n_input = 2
n_cell = 4
n_output = 4
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
bw_output=Output("bw_output", "TENSOR_FLOAT32", "{{{}, {}, {}}}".format(max_time, n_batch, n_output))

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

fw_input_to_forget_weights_data = [
    -0.55291498, -0.42866567, 0.13056988, -0.3633365,
    -0.22755712, 0.28253698, 0.24407166, 0.33826375
]
bw_input_to_forget_weights_data = fw_input_to_forget_weights_data

fw_input_to_cell_weights_data = [
    -0.49770179, -0.27711356, -0.09624726, 0.05100781,
    0.04717243, 0.48944736, -0.38535351, -0.17212132
]
bw_input_to_cell_weights_data = fw_input_to_cell_weights_data

fw_input_to_output_weights_data = [
    0.10725588, -0.02335852, -0.55932593, -0.09426838,
    -0.44257352, 0.54939759, 0.01533556, 0.42751634
]
bw_input_to_output_weights_data = fw_input_to_output_weights_data

fw_recurrent_to_forget_weights_data = [
    -0.13832897, -0.0515101, -0.2359007, -0.16661474,
    -0.14340827, 0.36986142, 0.23414481, 0.55899,
    0.10798943, -0.41174671, 0.17751795, -0.34484994,
    -0.35874045, -0.11352962, 0.27268326, 0.54058349
]
bw_recurrent_to_forget_weights_data = fw_recurrent_to_forget_weights_data

fw_recurrent_to_cell_weights_data = [
    0.54066205, -0.32668582, -0.43562764, -0.56094903,
    0.42957711, 0.01841056, -0.32764608, -0.33027974,
    -0.10826075, 0.20675004, 0.19069612, -0.03026325,
    -0.54532051, 0.33003211, 0.44901288, 0.21193194
]
bw_recurrent_to_cell_weights_data = fw_recurrent_to_cell_weights_data

fw_recurrent_to_output_weights_data = [
    0.41613156, 0.42610586, -0.16495961, -0.5663873,
    0.30579174, -0.05115908, -0.33941799, 0.23364776,
    0.11178309, 0.09481031, -0.26424935, 0.46261835,
    0.50248802, 0.26114327, -0.43736315, 0.33149987
]
bw_recurrent_to_output_weights_data = fw_recurrent_to_output_weights_data

fw_forget_gate_bias_data = [1.0, 1.0, 1.0, 1.0]
bw_forget_gate_bias_data = [1.0, 1.0, 1.0, 1.0]

fw_cell_bias_data = [0.0, 0.0, 0.0, 0.0]
bw_cell_bias_data = [0.0, 0.0, 0.0, 0.0]

fw_output_gate_bias_data = [0.0, 0.0, 0.0, 0.0]
bw_output_gate_bias_data = [0.0, 0.0, 0.0, 0.0]

fw_cell_to_forget_weights_data = [ 0.47485286, -0.51955009, -0.24458408, 0.31544167 ]
bw_cell_to_forget_weights_data = fw_cell_to_forget_weights_data

fw_cell_to_output_weights_data = [ -0.17135078, 0.82760304, 0.85573703, -0.77109635 ]
bw_cell_to_output_weights_data = fw_cell_to_output_weights_data

input_data = [2.0, 3.0, 3.0, 4.0, 1.0, 1.0]

fw_activation_state_data = [0 for _ in range(n_batch * n_output)]
bw_activation_state_data = [0 for _ in range(n_batch * n_output)]

fw_cell_state_data = [0 for _ in range(n_batch * n_cell)]
bw_cell_state_data = [0 for _ in range(n_batch * n_cell)]

fw_golden_output_data = [
    -0.36444446, -0.00352185, 0.12886585, -0.05163646,
    -0.42312205, -0.01218222, 0.24201041, -0.08124574,
    -0.358325,   -0.04621704, 0.21641694, -0.06471302
]
bw_golden_output_data = [
    -0.401685, -0.0232794,  0.288642,  -0.123074,
    -0.42915,  -0.00871577, 0.20912,   -0.103567,
    -0.166398, -0.00486649, 0.0697471, -0.0537578
]


test(
    name="blackbox",
    input_data=input_data,
    fw_input_to_forget_weights_data=fw_input_to_forget_weights_data,
    fw_input_to_cell_weights_data=fw_input_to_cell_weights_data,
    fw_input_to_output_weights_data=fw_input_to_output_weights_data,
    fw_recurrent_to_forget_weights_data=fw_recurrent_to_forget_weights_data,
    fw_recurrent_to_cell_weights_data=fw_recurrent_to_cell_weights_data,
    fw_recurrent_to_output_weights_data=fw_recurrent_to_output_weights_data,
    fw_cell_to_forget_weights_data = fw_cell_to_forget_weights_data,
    fw_cell_to_output_weights_data = fw_cell_to_output_weights_data,
    fw_forget_gate_bias_data=fw_forget_gate_bias_data,
    fw_cell_bias_data=fw_cell_bias_data,
    fw_output_gate_bias_data=fw_output_gate_bias_data,
    bw_input_to_forget_weights_data=bw_input_to_forget_weights_data,
    bw_input_to_cell_weights_data=bw_input_to_cell_weights_data,
    bw_input_to_output_weights_data=bw_input_to_output_weights_data,
    bw_recurrent_to_forget_weights_data=bw_recurrent_to_forget_weights_data,
    bw_recurrent_to_cell_weights_data=bw_recurrent_to_cell_weights_data,
    bw_recurrent_to_output_weights_data=bw_recurrent_to_output_weights_data,
    bw_cell_to_forget_weights_data = bw_cell_to_forget_weights_data,
    bw_cell_to_output_weights_data = bw_cell_to_output_weights_data,
    bw_forget_gate_bias_data=bw_forget_gate_bias_data,
    bw_cell_bias_data=bw_cell_bias_data,
    bw_output_gate_bias_data=bw_output_gate_bias_data,
    fw_activation_state_data = fw_activation_state_data,
    bw_activation_state_data = bw_activation_state_data,
    fw_cell_state_data = fw_cell_state_data,
    bw_cell_state_data = bw_cell_state_data,
    fw_output_data=fw_golden_output_data,
    bw_output_data=bw_golden_output_data
)
