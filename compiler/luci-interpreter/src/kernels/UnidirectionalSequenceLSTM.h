/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LUCI_INTERPRETER_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H
#define LUCI_INTERPRETER_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class UnidirectionalSequenceLSTM : public KernelWithParams<UnidirectionalSequenceLSTMParams>
{
public:
  UnidirectionalSequenceLSTM(
    const Tensor *input, const Tensor *input_to_input_weights,
    const Tensor *input_to_forget_weights, const Tensor *input_to_cell_weights,
    const Tensor *input_to_output_weights, const Tensor *recurrent_to_input_weights,
    const Tensor *recurrent_to_forget_weights, const Tensor *recurrent_to_cell_weights,
    const Tensor *recurrent_to_output_weights, const Tensor *cell_to_input_weights,
    const Tensor *cell_to_forget_weights, const Tensor *cell_to_output_weights,
    const Tensor *input_gate_bias, const Tensor *forget_gate_bias, const Tensor *cell_gate_bias,
    const Tensor *output_gate_bias, const Tensor *projection_weights, const Tensor *projection_bias,
    const Tensor *activation_state, const Tensor *cell_state,
    const Tensor *input_layer_norm_coefficients, const Tensor *forget_layer_norm_coefficients,
    const Tensor *cell_layer_norm_coefficients, const Tensor *output_layer_norm_coefficients,
    std::vector<Tensor *> &&outputs, const UnidirectionalSequenceLSTMParams &params);

  const Tensor *input() const { return _inputs[0]; }
  const Tensor *input_to_input_weights() const { return _inputs[1]; }
  const Tensor *input_to_forget_weights() const { return _inputs[2]; }
  const Tensor *input_to_cell_weights() const { return _inputs[3]; }
  const Tensor *input_to_output_weights() const { return _inputs[4]; }
  const Tensor *recurrent_to_input_weights() const { return _inputs[5]; }
  const Tensor *recurrent_to_forget_weights() const { return _inputs[6]; }
  const Tensor *recurrent_to_cell_weights() const { return _inputs[7]; }
  const Tensor *recurrent_to_output_weights() const { return _inputs[8]; }
  const Tensor *cell_to_input_weights() const { return _inputs[9]; }
  const Tensor *cell_to_forget_weights() const { return _inputs[10]; }
  const Tensor *cell_to_output_weights() const { return _inputs[11]; }
  const Tensor *input_gate_bias() const { return _inputs[12]; }
  const Tensor *forget_gate_bias() const { return _inputs[13]; }
  const Tensor *cell_gate_bias() const { return _inputs[14]; }
  const Tensor *output_gate_bias() const { return _inputs[15]; }
  const Tensor *projection_weights() const { return _inputs[16]; }
  const Tensor *projection_bias() const { return _inputs[17]; }
  const Tensor *output_state() const { return _inputs[18]; }
  const Tensor *cell_state() const { return _inputs[19]; }
  const Tensor *input_layer_norm_coefficients() const { return _inputs[20]; }
  const Tensor *forget_layer_norm_coefficients() const { return _inputs[21]; }
  const Tensor *cell_layer_norm_coefficients() const { return _inputs[22]; }
  const Tensor *output_layer_norm_coefficients() const { return _inputs[23]; }

  Tensor *output() const { return _outputs[0]; }
  Tensor *scratch_buffer() const { return _outputs[1]; }

  // Is hybrid scratchpad tensors
  Tensor *input_quantized() const { return _outputs.size() == 13 ? _outputs[2] : nullptr; }
  Tensor *output_state_quantized() const { return _outputs.size() == 13 ? _outputs[3] : nullptr; }
  Tensor *cell_state_quantized() const { return _outputs.size() == 13 ? _outputs[4] : nullptr; }
  Tensor *input_sf() const { return _outputs.size() == 13 ? _outputs[5] : nullptr; }
  Tensor *output_state_sf() const { return _outputs.size() == 13 ? _outputs[6] : nullptr; }
  Tensor *prod_scaling_factors() const { return _outputs.size() == 13 ? _outputs[7] : nullptr; }
  Tensor *recovered_cell_weights() const { return _outputs.size() == 13 ? _outputs[8] : nullptr; }
  Tensor *accum_scratch() const { return _outputs.size() == 13 ? _outputs[9] : nullptr; }
  Tensor *input_zp() const { return _outputs.size() == 13 ? _outputs[10] : nullptr; }
  Tensor *output_state_zp() const { return _outputs.size() == 13 ? _outputs[11] : nullptr; }
  Tensor *row_sums() const { return _outputs.size() == 13 ? _outputs[12] : nullptr; }

  void configure() override;
  void execute() const override;

private:
  void checkInputTensorDimensions(int n_input, int n_output, int n_cell, bool is_integer) const;

  void evalFloat() const;
  void evalHybrid() const;

  bool _compute_row_sums = false;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_H
