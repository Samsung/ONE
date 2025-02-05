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
# FLOAT16, Batch Major, Aux Input, No Layer Normalization, No Cifg, No Peephole, No Projection,
# and No Clipping.
#
# Adapted from TFLite's LSTMOpTest.BlackBoxTestWithAuxInput.

n_batch = 1
n_input = 2
n_cell = 4
n_output = 4
max_time = 3

input = Input("input", "TENSOR_FLOAT16", "{{{}, {}, {}}}".format(n_batch, max_time, n_input))

fw_input_to_input_weights = Input(
    "fw_input_to_input_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
fw_input_to_forget_weights = Input(
    "fw_input_to_forget_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
fw_input_to_cell_weights = Input(
    "fw_input_to_cell_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
fw_input_to_output_weights = Input(
    "fw_input_to_output_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))

fw_recurrent_to_input_weights = Input(
    "fw_recurrent_to_input_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))
fw_recurrent_to_forget_weights = Input(
    "fw_recurrent_to_forget_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))
fw_recurrent_to_cell_weights = Input(
    "fw_recurrent_to_cell_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))
fw_recurrent_to_output_weights = Input(
    "fw_recurrent_to_output_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))

fw_cell_to_input_weights = Input(
    "fw_cell_to_input_weights", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
fw_cell_to_forget_weights = Input(
    "fw_cell_to_forget_weights", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
fw_cell_to_output_weights = Input(
    "fw_cell_to_output_weights", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))

fw_input_gate_bias = Input(
    "fw_input_gate_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
fw_forget_gate_bias = Input(
    "fw_forget_gate_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
fw_cell_bias = Input(
    "fw_cell_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
fw_output_gate_bias = Input(
    "fw_output_gate_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))

fw_projection_weights = Input(
    "fw_projection_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_output, n_cell))
fw_projection_bias = Input(
    "fw_projection_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_output))

bw_input_to_input_weights = Input(
    "bw_input_to_input_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
bw_input_to_forget_weights = Input(
    "bw_input_to_forget_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
bw_input_to_cell_weights = Input(
    "bw_input_to_cell_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
bw_input_to_output_weights = Input(
    "bw_input_to_output_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))

bw_recurrent_to_input_weights = Input(
    "bw_recurrent_to_input_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))
bw_recurrent_to_forget_weights = Input(
    "bw_recurrent_to_forget_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))
bw_recurrent_to_cell_weights = Input(
    "bw_recurrent_to_cell_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))
bw_recurrent_to_output_weights = Input(
    "bw_recurrent_to_output_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_output))

bw_cell_to_input_weights = Input(
    "bw_cell_to_input_weights", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
bw_cell_to_forget_weights = Input(
    "bw_cell_to_forget_weights", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
bw_cell_to_output_weights = Input(
    "bw_cell_to_output_weights", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))

bw_input_gate_bias = Input(
    "bw_input_gate_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
bw_forget_gate_bias = Input(
    "bw_forget_gate_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
bw_cell_bias = Input(
    "bw_cell_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))
bw_output_gate_bias = Input(
    "bw_output_gate_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_cell))

bw_projection_weights = Input(
    "bw_projection_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_output, n_cell))
bw_projection_bias = Input(
    "bw_projection_bias", "TENSOR_FLOAT16", "{{{}}}".format(n_output))

fw_activation_state = Input(
    "fw_activatiom_state", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_batch, n_output))
fw_cell_state = Input(
    "fw_cell_state", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_batch, n_cell))

bw_activation_state = Input(
    "bw_activatiom_state", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_batch, n_output))
bw_cell_state = Input(
    "bw_cell_state", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_batch, n_cell))

aux_input = Input("input", "TENSOR_FLOAT16", "{{{}, {}, {}}}".format(n_batch, max_time, n_input))

fw_aux_input_to_input_weights = Input(
    "fw_aux_input_to_input_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
fw_aux_input_to_forget_weights = Input(
    "fw_input_to_forget_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
fw_aux_input_to_cell_weights = Input(
    "fw_aux_input_to_cell_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
fw_aux_input_to_output_weights = Input(
    "fw_aux_input_to_output_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))

bw_aux_input_to_input_weights = Input(
    "bw_aux_input_to_input_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
bw_aux_input_to_forget_weights = Input(
    "bw_input_to_forget_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
bw_aux_input_to_cell_weights = Input(
    "bw_aux_input_to_cell_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))
bw_aux_input_to_output_weights = Input(
    "bw_aux_input_to_output_weights", "TENSOR_FLOAT16", "{{{}, {}}}".format(n_cell, n_input))

fw_input_layer_norm_weights = Input("input_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)
fw_forget_layer_norm_weights = Input("forget_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)
fw_cell_layer_norm_weights = Input("cell_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)
fw_output_layer_norm_weights = Input("output_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)

bw_input_layer_norm_weights = Input("input_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)
bw_forget_layer_norm_weights = Input("forget_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)
bw_cell_layer_norm_weights = Input("cell_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)
bw_output_layer_norm_weights = Input("output_layer_norm_weights", "TENSOR_FLOAT16", "{%d}" % n_cell)

fw_output=Output("fw_output", "TENSOR_FLOAT16", "{{{}, {}, {}}}".format(n_batch, max_time, n_output))
bw_output=Output("bw_output", "TENSOR_FLOAT16", "{{{}, {}, {}}}".format(n_batch, max_time, n_output))

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

  activation = Int32Scalar("activation", 4) # Tanh
  cell_clip = Float16Scalar("cell_clip", 0.0)
  proj_clip = Float16Scalar("proj_clip", 0.0)
  merge_outputs = BoolScalar("merge_outputs", False)
  time_major = BoolScalar("time_major", False)

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
    -0.45018822, -0.02338299, -0.0870589,
    -0.34550029, 0.04266912, -0.15680569,
    -0.34856534, 0.43890524
]
bw_input_to_input_weights_data = fw_input_to_input_weights_data

fw_input_to_forget_weights_data = [
    0.09701663, 0.20334584, -0.50592935,
    -0.31343272, -0.40032279, 0.44781327,
    0.01387155, -0.35593212
]
bw_input_to_forget_weights_data = fw_input_to_forget_weights_data

fw_input_to_cell_weights_data = [
    -0.50013041, 0.1370284, 0.11810488, 0.2013163,
    -0.20583314, 0.44344562, 0.22077113,
    -0.29909778
]
bw_input_to_cell_weights_data = fw_input_to_cell_weights_data

fw_input_to_output_weights_data = [
    -0.25065863, -0.28290087, 0.04613829,
    0.40525138, 0.44272184, 0.03897077, -0.1556896,
    0.19487578
]
bw_input_to_output_weights_data = fw_input_to_output_weights_data

fw_recurrent_to_input_weights_data = [
    -0.0063535, -0.2042388, 0.31454784, -0.35746509, 0.28902304, 0.08183324,
    -0.16555229, 0.02286911, -0.13566875, 0.03034258, 0.48091322,
    -0.12528998, 0.24077177, -0.51332325, -0.33502164, 0.10629296
]
bw_recurrent_to_input_weights_data = fw_recurrent_to_input_weights_data

fw_recurrent_to_forget_weights_data = [
    -0.48684245, -0.06655136, 0.42224967, 0.2112639, 0.27654213, 0.20864892,
    -0.07646349, 0.45877004, 0.00141793, -0.14609534, 0.36447752, 0.09196436,
    0.28053468, 0.01560611, -0.20127171, -0.01140004
]
bw_recurrent_to_forget_weights_data = fw_recurrent_to_forget_weights_data

fw_recurrent_to_cell_weights_data = [
    -0.3407414, 0.24443203, -0.2078532, 0.26320225, 0.05695659, -0.00123841,
    -0.4744786, -0.35869038, -0.06418842, -0.13502428, -0.501764, 0.22830659,
    -0.46367589, 0.26016325, -0.03894562, -0.16368064
]
bw_recurrent_to_cell_weights_data = fw_recurrent_to_cell_weights_data

fw_recurrent_to_output_weights_data = [
    0.43385774, -0.17194885, 0.2718237, 0.09215671, 0.24107647, -0.39835793,
    0.18212086, 0.01301402, 0.48572797, -0.50656658, 0.20047462, -0.20607421,
    -0.51818722, -0.15390486, 0.0468148, 0.39922136
]
bw_recurrent_to_output_weights_data = fw_recurrent_to_output_weights_data

fw_input_gate_bias_data = [0.0, 0.0, 0.0, 0.0]
bw_input_gate_bias_data = fw_input_gate_bias_data

fw_forget_gate_bias_data = [1.0, 1.0, 1.0, 1.0]
bw_forget_gate_bias_data = fw_forget_gate_bias_data

fw_cell_bias_data = [0.0, 0.0, 0.0, 0.0]
bw_cell_bias_data = fw_cell_bias_data

fw_output_gate_bias_data = [0.0, 0.0, 0.0, 0.0]
bw_output_gate_bias_data = fw_output_gate_bias_data

input_data = [2.0, 3.0, 3.0, 4.0, 1.0, 1.0]
aux_input_data = input_data

fw_aux_input_to_input_weights_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
bw_aux_input_to_input_weights_data = fw_aux_input_to_input_weights_data
fw_aux_input_to_forget_weights_data = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0]
bw_aux_input_to_forget_weights_data = fw_aux_input_to_forget_weights_data
fw_aux_input_to_cell_weights_data = [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8]
bw_aux_input_to_cell_weights_data = fw_aux_input_to_cell_weights_data
fw_aux_input_to_output_weights_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
bw_aux_input_to_output_weights_data = fw_aux_input_to_output_weights_data

fw_activation_state_data = [0 for _ in range(n_batch * n_output)]
bw_activation_state_data = [0 for _ in range(n_batch * n_output)]

fw_cell_state_data = [0 for _ in range(n_batch * n_cell)]
bw_cell_state_data = [0 for _ in range(n_batch * n_cell)]

fw_golden_output_data = [
    0.153335, 0.542754, 0.708602, 0.742855,
    0.247581, 0.835739, 0.947797, 0.958177,
    0.410892, 0.672268, 0.761909, 0.829133
]
bw_golden_output_data = [
    0.342275, 0.883431, 0.955930, 0.975621,
    0.204939, 0.806858, 0.914849, 0.934871,
    0.123236, 0.373087, 0.465377, 0.517630
]


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
    fw_input_gate_bias_data=fw_input_gate_bias_data,
    fw_forget_gate_bias_data=fw_forget_gate_bias_data,
    fw_cell_bias_data=fw_cell_bias_data,
    fw_output_gate_bias_data=fw_output_gate_bias_data,
    bw_input_to_input_weights_data=bw_input_to_input_weights_data,
    bw_input_to_forget_weights_data=bw_input_to_forget_weights_data,
    bw_input_to_cell_weights_data=bw_input_to_cell_weights_data,
    bw_input_to_output_weights_data=bw_input_to_output_weights_data,
    bw_recurrent_to_input_weights_data=bw_recurrent_to_input_weights_data,
    bw_recurrent_to_forget_weights_data=bw_recurrent_to_forget_weights_data,
    bw_recurrent_to_cell_weights_data=bw_recurrent_to_cell_weights_data,
    bw_recurrent_to_output_weights_data=bw_recurrent_to_output_weights_data,
    bw_input_gate_bias_data=bw_input_gate_bias_data,
    bw_forget_gate_bias_data=bw_forget_gate_bias_data,
    bw_cell_bias_data=bw_cell_bias_data,
    bw_output_gate_bias_data=bw_output_gate_bias_data,
    fw_activation_state_data = fw_activation_state_data,
    bw_activation_state_data = bw_activation_state_data,
    fw_cell_state_data = fw_cell_state_data,
    bw_cell_state_data = bw_cell_state_data,
    aux_input_data = aux_input_data,
    fw_aux_input_to_input_weights_data=fw_aux_input_to_input_weights_data,
    fw_aux_input_to_forget_weights_data=fw_aux_input_to_forget_weights_data,
    fw_aux_input_to_cell_weights_data=fw_aux_input_to_cell_weights_data,
    fw_aux_input_to_output_weights_data=fw_aux_input_to_output_weights_data,
    bw_aux_input_to_input_weights_data=bw_aux_input_to_input_weights_data,
    bw_aux_input_to_forget_weights_data=bw_aux_input_to_forget_weights_data,
    bw_aux_input_to_cell_weights_data=bw_aux_input_to_cell_weights_data,
    bw_aux_input_to_output_weights_data=bw_aux_input_to_output_weights_data,
    fw_output_data=fw_golden_output_data,
    bw_output_data=bw_golden_output_data
)
