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

#ifndef LUCI_INTERPRETER_KERNELS_BIDIRECTIONALSEQUENCELSTM_H
#define LUCI_INTERPRETER_KERNELS_BIDIRECTIONALSEQUENCELSTM_H

#include "core/Kernel.h"
#include "core/KernelParams.h"

namespace luci_interpreter
{
namespace kernels
{

class BidirectionalSequenceLSTM : public KernelWithParams<BidirectionalSequenceLSTMParams>
{
public:
  BidirectionalSequenceLSTM(
    const Tensor *input,

    const Tensor *fw_input_to_input_weights, const Tensor *fw_input_to_forget_weights,
    const Tensor *fw_input_to_cell_weights, const Tensor *fw_input_to_output_weights,

    const Tensor *fw_recurrent_to_input_weights, const Tensor *fw_recurrent_to_forget_weights,
    const Tensor *fw_recurrent_to_cell_weights, const Tensor *fw_recurrent_to_output_weights,

    const Tensor *fw_cell_to_input_weights, const Tensor *fw_cell_to_forget_weights,
    const Tensor *fw_cell_to_output_weights,

    const Tensor *fw_input_gate_bias, const Tensor *fw_forget_gate_bias,
    const Tensor *fw_cell_gate_bias, const Tensor *fw_output_gate_bias,

    const Tensor *fw_projection_weights, const Tensor *fw_projection_bias,

    const Tensor *bw_input_to_input_weights, const Tensor *bw_input_to_forget_weights,
    const Tensor *bw_input_to_cell_weights, const Tensor *bw_input_to_output_weights,

    const Tensor *bw_recurrent_to_input_weights, const Tensor *bw_recurrent_to_forget_weights,
    const Tensor *bw_recurrent_to_cell_weights, const Tensor *bw_recurrent_to_output_weights,

    const Tensor *bw_cell_to_input_weights, const Tensor *bw_cell_to_forget_weights,
    const Tensor *bw_cell_to_output_weights,

    const Tensor *bw_input_gate_bias, const Tensor *bw_forget_gate_bias,
    const Tensor *bw_cell_gate_bias, const Tensor *bw_output_gate_bias,

    const Tensor *bw_projection_weights, const Tensor *bw_projection_bias,

    const Tensor *fw_input_activation_state, const Tensor *fw_input_cell_state,

    const Tensor *bw_input_activation_state, const Tensor *bw_input_cell_state,

    const Tensor *aux_input,

    const Tensor *fw_aux_input_to_input_weights, const Tensor *fw_aux_input_to_forget_weights,
    const Tensor *fw_aux_input_to_cell_weights, const Tensor *fw_aux_input_to_output_weights,

    const Tensor *bw_aux_input_to_input_weights, const Tensor *bw_aux_input_to_forget_weights,
    const Tensor *bw_aux_input_to_cell_weights, const Tensor *bw_aux_input_to_output_weights,

    Tensor *fw_output, Tensor *bw_output,

    Tensor *fw_scratchpad, Tensor *bw_scratchpad,

    const BidirectionalSequenceLSTMParams &params);

  const Tensor *input() const { return _inputs[0]; }

  const Tensor *fw_input_to_input_weights() const { return _inputs[1]; }
  const Tensor *fw_input_to_forget_weights() const { return _inputs[2]; }
  const Tensor *fw_input_to_cell_weights() const { return _inputs[3]; }
  const Tensor *fw_input_to_output_weights() const { return _inputs[4]; }

  const Tensor *fw_recurrent_to_input_weights() const { return _inputs[5]; }
  const Tensor *fw_recurrent_to_forget_weights() const { return _inputs[6]; }
  const Tensor *fw_recurrent_to_cell_weights() const { return _inputs[7]; }
  const Tensor *fw_recurrent_to_output_weights() const { return _inputs[8]; }

  const Tensor *fw_cell_to_input_weights() const { return _inputs[9]; }
  const Tensor *fw_cell_to_forget_weights() const { return _inputs[10]; }
  const Tensor *fw_cell_to_output_weights() const { return _inputs[11]; }

  const Tensor *fw_input_gate_bias() const { return _inputs[12]; }
  const Tensor *fw_forget_gate_bias() const { return _inputs[13]; }
  const Tensor *fw_cell_gate_bias() const { return _inputs[14]; }
  const Tensor *fw_output_gate_bias() const { return _inputs[15]; }

  const Tensor *fw_projection_weights() const { return _inputs[16]; }
  const Tensor *fw_projection_bias() const { return _inputs[17]; }

  const Tensor *bw_input_to_input_weights() const { return _inputs[18]; }
  const Tensor *bw_input_to_forget_weights() const { return _inputs[19]; }
  const Tensor *bw_input_to_cell_weights() const { return _inputs[20]; }
  const Tensor *bw_input_to_output_weights() const { return _inputs[21]; }

  const Tensor *bw_recurrent_to_input_weights() const { return _inputs[22]; }
  const Tensor *bw_recurrent_to_forget_weights() const { return _inputs[23]; }
  const Tensor *bw_recurrent_to_cell_weights() const { return _inputs[24]; }
  const Tensor *bw_recurrent_to_output_weights() const { return _inputs[25]; }

  const Tensor *bw_cell_to_input_weights() const { return _inputs[26]; }
  const Tensor *bw_cell_to_forget_weights() const { return _inputs[27]; }
  const Tensor *bw_cell_to_output_weights() const { return _inputs[28]; }

  const Tensor *bw_input_gate_bias() const { return _inputs[29]; }
  const Tensor *bw_forget_gate_bias() const { return _inputs[30]; }
  const Tensor *bw_cell_gate_bias() const { return _inputs[31]; }
  const Tensor *bw_output_gate_bias() const { return _inputs[32]; }

  const Tensor *bw_projection_weights() const { return _inputs[33]; }
  const Tensor *bw_projection_bias() const { return _inputs[34]; }

  const Tensor *fw_input_activation_state() const { return _inputs[35]; }
  const Tensor *fw_input_cell_state() const { return _inputs[36]; }

  const Tensor *bw_input_activation_state() const { return _inputs[37]; }
  const Tensor *bw_input_cell_state() const { return _inputs[38]; }

  const Tensor *aux_input() const { return _inputs[39]; }

  const Tensor *fw_aux_input_to_input_weights() const { return _inputs[40]; }
  const Tensor *fw_aux_input_to_forget_weights() const { return _inputs[41]; }
  const Tensor *fw_aux_input_to_cell_weights() const { return _inputs[42]; }
  const Tensor *fw_aux_input_to_output_weights() const { return _inputs[43]; }

  const Tensor *bw_aux_input_to_input_weights() const { return _inputs[44]; }
  const Tensor *bw_aux_input_to_forget_weights() const { return _inputs[45]; }
  const Tensor *bw_aux_input_to_cell_weights() const { return _inputs[46]; }
  const Tensor *bw_aux_input_to_output_weights() const { return _inputs[47]; }

  Tensor *fw_output() const { return _outputs[0]; }
  Tensor *bw_output() const { return _outputs[1]; }

  void configure() override;
  void execute() const override;

private:
  void evalFloat() const;

private:
  void check_input_tensor_dimensions(
    int n_input, int n_input, int n_output, int n_cell, int input_to_input_weights_tensor,
    int input_to_forget_weights_tensor, int input_to_cell_weights_tensor,
    int input_to_output_weights_tensor, int recurrent_to_input_weights_tensor,
    int recurrent_to_forget_weights_tensor, int recurrent_to_cell_weights_tensor,
    int recurrent_to_output_weights_tensor, int cell_to_input_weights_tensor,
    int cell_to_forget_weights_tensor, int cell_to_output_weights_tensor,
    int input_gate_bias_tensor, int forget_gate_bias_tensor, int cell_gate_bias_tensor,
    int output_gate_bias_tensor, int projection_weights_tensor, int projection_bias_tensor);
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_BIDIRECTIONALSEQUENCELSTM_H
