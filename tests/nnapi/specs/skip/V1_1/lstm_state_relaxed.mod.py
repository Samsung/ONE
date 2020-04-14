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

# LSTM Test: No Cifg, No Peephole, No Projection, and No Clipping.

model = Model()

n_batch = 1
n_input = 2
# n_cell and n_output have the same size when there is no projection.
n_cell = 4
n_output = 4

input = Input("input", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_input))

input_to_input_weights = Input("input_to_input_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_input))
input_to_forget_weights = Input("input_to_forget_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_input))
input_to_cell_weights = Input("input_to_cell_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_input))
input_to_output_weights = Input("input_to_output_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_input))

recurrent_to_input_weights = Input("recurrent_to_intput_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_output))
recurrent_to_forget_weights = Input("recurrent_to_forget_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_output))
recurrent_to_cell_weights = Input("recurrent_to_cell_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_output))
recurrent_to_output_weights = Input("recurrent_to_output_weights", "TENSOR_FLOAT32", "{%d, %d}" % (n_cell, n_output))

cell_to_input_weights = Input("cell_to_input_weights", "TENSOR_FLOAT32", "{0}")
cell_to_forget_weights = Input("cell_to_forget_weights", "TENSOR_FLOAT32", "{0}")
cell_to_output_weights = Input("cell_to_output_weights", "TENSOR_FLOAT32", "{0}")

input_gate_bias = Input("input_gate_bias", "TENSOR_FLOAT32", "{%d}"%(n_cell))
forget_gate_bias = Input("forget_gate_bias", "TENSOR_FLOAT32", "{%d}"%(n_cell))
cell_gate_bias = Input("cell_gate_bias", "TENSOR_FLOAT32", "{%d}"%(n_cell))
output_gate_bias = Input("output_gate_bias", "TENSOR_FLOAT32", "{%d}"%(n_cell))

projection_weights = Input("projection_weights", "TENSOR_FLOAT32", "{0,0}")
projection_bias = Input("projection_bias", "TENSOR_FLOAT32", "{0}")

output_state_in = Input("output_state_in", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_output))
cell_state_in = Input("cell_state_in", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_cell))

activation_param = Int32Scalar("activation_param", 4)  # Tanh
cell_clip_param = Float32Scalar("cell_clip_param", 0.)
proj_clip_param = Float32Scalar("proj_clip_param", 0.)

scratch_buffer = IgnoredOutput("scratch_buffer", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, (n_cell * 4)))
output_state_out = Output("output_state_out", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_output))
cell_state_out = Output("cell_state_out", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_cell))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (n_batch, n_output))

model = model.Operation("LSTM",
                        input,

                        input_to_input_weights,
                        input_to_forget_weights,
                        input_to_cell_weights,
                        input_to_output_weights,

                        recurrent_to_input_weights,
                        recurrent_to_forget_weights,
                        recurrent_to_cell_weights,
                        recurrent_to_output_weights,

                        cell_to_input_weights,
                        cell_to_forget_weights,
                        cell_to_output_weights,

                        input_gate_bias,
                        forget_gate_bias,
                        cell_gate_bias,
                        output_gate_bias,

                        projection_weights,
                        projection_bias,

                        output_state_in,
                        cell_state_in,

                        activation_param,
                        cell_clip_param,
                        proj_clip_param
).To([scratch_buffer, output_state_out, cell_state_out, output])
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {input_to_input_weights:  [-0.45018822, -0.02338299, -0.0870589, -0.34550029, 0.04266912, -0.15680569, -0.34856534, 0.43890524],
          input_to_forget_weights: [0.09701663, 0.20334584, -0.50592935, -0.31343272, -0.40032279, 0.44781327, 0.01387155, -0.35593212],
          input_to_cell_weights:   [-0.50013041, 0.1370284, 0.11810488, 0.2013163, -0.20583314, 0.44344562, 0.22077113, -0.29909778],
          input_to_output_weights: [-0.25065863, -0.28290087, 0.04613829, 0.40525138, 0.44272184, 0.03897077, -0.1556896, 0.19487578],

          input_gate_bias:  [0.,0.,0.,0.],
          forget_gate_bias: [1.,1.,1.,1.],
          cell_gate_bias:   [0.,0.,0.,0.],
          output_gate_bias: [0.,0.,0.,0.],

          recurrent_to_input_weights: [
              -0.0063535, -0.2042388, 0.31454784, -0.35746509, 0.28902304, 0.08183324,
            -0.16555229, 0.02286911, -0.13566875, 0.03034258, 0.48091322,
            -0.12528998, 0.24077177, -0.51332325, -0.33502164, 0.10629296],

          recurrent_to_cell_weights: [
              -0.3407414, 0.24443203, -0.2078532, 0.26320225, 0.05695659, -0.00123841,
            -0.4744786, -0.35869038, -0.06418842, -0.13502428, -0.501764, 0.22830659,
            -0.46367589, 0.26016325, -0.03894562, -0.16368064],

          recurrent_to_forget_weights: [
              -0.48684245, -0.06655136, 0.42224967, 0.2112639, 0.27654213, 0.20864892,
            -0.07646349, 0.45877004, 0.00141793, -0.14609534, 0.36447752, 0.09196436,
            0.28053468, 0.01560611, -0.20127171, -0.01140004],

          recurrent_to_output_weights: [
              0.43385774, -0.17194885, 0.2718237, 0.09215671, 0.24107647, -0.39835793,
              0.18212086, 0.01301402, 0.48572797, -0.50656658, 0.20047462, -0.20607421,
              -0.51818722, -0.15390486, 0.0468148, 0.39922136],

          cell_to_input_weights: [],
          cell_to_forget_weights: [],
          cell_to_output_weights: [],

          projection_weights: [],
          projection_bias: [],
}

test_input = [3., 4.]
output_state = [-0.0297319, 0.122947, 0.208851, -0.153588]
cell_state = [-0.145439, 0.157475, 0.293663, -0.277353,]
golden_output = [-0.03716109, 0.12507336, 0.41193449,  -0.20860538]
output0 = {
    scratch_buffer: [ 0 for x in range(n_batch * n_cell * 4) ],
    cell_state_out: [ -0.287121, 0.148115, 0.556837, -0.388276 ],
    output_state_out: [ -0.0371611, 0.125073, 0.411934, -0.208605 ],
    output: golden_output
}
input0[input] = test_input
input0[output_state_in] = output_state
input0[cell_state_in] = cell_state
Example((input0, output0))
