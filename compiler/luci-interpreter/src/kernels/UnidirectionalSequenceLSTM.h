/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_KERNELS_UNIDIRECTIONALSEQUENCELSTM_H
#define LUCI_INTERPRETER_KERNELS_UNIDIRECTIONALSEQUENCELSTM_H

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
    const Tensor *input,

    const Tensor *input_to_input_weights, const Tensor *input_to_forget_weights,
    const Tensor *input_to_cell_weights, const Tensor *input_to_output_weights,

    const Tensor *recurrent_to_input_weights, const Tensor *recurrent_to_forget_weights,
    const Tensor *recurrent_to_cell_weights, const Tensor *recurrent_to_output_weights,

    const Tensor *cell_to_input_weights, const Tensor *cell_to_forget_weights,
    const Tensor *cell_to_output_weights,

    const Tensor *input_gate_bias, const Tensor *forget_gate_bias, const Tensor *cell_gate_bias,
    const Tensor *output_gate_bias,

    const Tensor *projection_weights, const Tensor *projection_bias,

    const Tensor *output_state, const Tensor *cell_state,

    const Tensor *input_layer_norm_coefficients, const Tensor *forget_layer_norm_coefficients,
    const Tensor *cell_layer_norm_coefficients, const Tensor *output_layer_norm_coefficients,

    Tensor *output, Tensor *scratchpad_1, Tensor *scratchpad_2, Tensor *scratchpad_3,
    const UnidirectionalSequenceLSTMParams &params);

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

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;

private:
  void check_input_tensor_dimensions(int n_input, int n_output, int n_cell, bool use_layer_norm,
                                     bool is_integer);
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UNIDIRECTIONALSEQUENCELSTM_H
