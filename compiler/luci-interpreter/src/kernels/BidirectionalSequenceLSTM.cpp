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

#include "kernels/BidirectionalSequenceLSTM.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/tensor_utils.h>

namespace luci_interpreter
{
namespace kernels
{

BidirectionalSequenceLSTM::BidirectionalSequenceLSTM(
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

  const BidirectionalSequenceLSTMParams &params)
  : KernelWithParams<BidirectionalSequenceLSTMParams>(
      input,

      fw_input_to_input_weights, fw_input_to_forget_weights, fw_input_to_cell_weights,
      fw_input_to_output_weights,

      fw_recurrent_to_input_weights, fw_recurrent_to_forget_weights, fw_recurrent_to_cell_weights,
      fw_recurrent_to_output_weights,

      fw_cell_to_input_weights, fw_cell_to_forget_weights, fw_cell_to_output_weights,

      fw_input_gate_bias, fw_forget_gate_bias, fw_cell_gate_bias, fw_output_gate_bias,

      fw_projection_weights, fw_projection_bias,

      bw_input_to_input_weights, bw_input_to_forget_weights, bw_input_to_cell_weights,
      bw_input_to_output_weights,

      bw_recurrent_to_input_weights, bw_recurrent_to_forget_weights, bw_recurrent_to_cell_weights,
      bw_recurrent_to_output_weights,

      bw_cell_to_input_weights, bw_cell_to_forget_weights, bw_cell_to_output_weights,

      bw_input_gate_bias, bw_forget_gate_bias, bw_cell_gate_bias, bw_output_gate_bias,

      bw_projection_weights, bw_projection_bias,

      fw_input_activation_state, fw_input_cell_state,

      bw_input_activation_state, bw_input_cell_state,

      aux_input,

      fw_aux_input_to_input_weights, fw_aux_input_to_forget_weights, fw_aux_input_to_cell_weights,
      fw_aux_input_to_output_weights,

      bw_aux_input_to_input_weights, bw_aux_input_to_forget_weights, bw_aux_input_to_cell_weights,
      bw_aux_input_to_output_weights,

      {fw_output, bw_output, fw_scratchpad, bw_scratchpad}, params)
{
  // Do nothing
}

void BidirectionalSequenceLSTM::configure()
{
  LUCI_INTERPRETER_CHECK(getInputTensors().size() == 48);
  LUCI_INTERPRETER_CHECK(getOutputTensors().size() >= 1);

  // TODO implement
}

void BidirectionalSequenceLSTM::execute() const
{
  switch (input()->element_type())
  {
    case loco::DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type");
  }
}

void BidirectionalSequenceLSTM::evalFloat() const
{
  const bool time_major = params().time_major;
  const bool use_layer_norm = (forget_layer_norm_coefficients() != nullptr);

  const Tensor *t_fw_input_layer_norm_coefficients =
    use_layer_norm ? fw_input_layer_norm_coefficients() : nullptr;
  const Tensor *t_fw_forget_layer_norm_coefficients =
    use_layer_norm ? fw_forget_layer_norm_coefficients() : nullptr;
  const Tensor *t_fw_cell_layer_norm_coefficients =
    use_layer_norm ? fw_cell_layer_norm_coefficients() : nullptr;
  const Tensor *t_fw_output_layer_norm_coefficients =
    use_layer_norm ? fw_output_layer_norm_coefficients() : nullptr;

  const Tensor *t_bw_input_layer_norm_coefficients =
    use_layer_norm ? bw_input_layer_norm_coefficients() : nullptr;
  const Tensor *t_bw_forget_layer_norm_coefficients =
    use_layer_norm ? bw_forget_layer_norm_coefficients() : nullptr;
  const Tensor *t_bw_cell_layer_norm_coefficients =
    use_layer_norm ? bw_cell_layer_norm_coefficients() : nullptr;
  const Tensor *t_bw_output_layer_norm_coefficients =
    use_layer_norm ? bw_output_layer_norm_coefficients() : nullptr;

  Tensor *sp_output_state = getOutputTensors()[1];
  Tensor *sp_cell_state = getOutputTensors()[2];
  Tensor *sp_scratch_buffer = getOutputTensors()[3];

  // Note: it is expected that output_state input variable tensor reset to zero,
  // also expected that this variable tensor doesn't have buffer
  auto scratchpad_data = getTensorData<float>(sp_output_state);
  std::fill_n(scratchpad_data, sp_output_state->shape().num_elements(), 0);
  scratchpad_data = getTensorData<float>(sp_cell_state);
  std::fill_n(scratchpad_data, sp_cell_state->shape().num_elements(), 0);
  scratchpad_data = getTensorData<float>(sp_scratch_buffer);
  std::fill_n(scratchpad_data, sp_scratch_buffer->shape().num_elements(), 0);

  TfLiteLSTMParams lstm_params{};
  lstm_params.activation = getTfLiteActivation(params().activation);
  lstm_params.cell_clip = params().cell_clip;
  lstm_params.proj_clip = params().proj_clip;
  lstm_params.asymmetric_quantize_inputs = params().asymmetric_quantize_inputs;

  lstm::EvalFloat(
    input(), fw_input_to_input_weights(), fw_input_to_forget_weights(), fw_input_to_cell_weights(),
    fw_input_to_output_weights(),

    fw_recurrent_to_input_weights(), fw_recurrent_to_forget_weights(),
    fw_recurrent_to_cell_weights(), fw_recurrent_to_output_weights(),

    fw_cell_to_input_weights(), fw_cell_to_forget_weights(), fw_cell_to_output_weights(),

    t_fw_input_layer_norm_coefficients, t_fw_forget_layer_norm_coefficients,
    t_fw_cell_layer_norm_coefficients, t_fw_output_layer_norm_coefficients,
    /*aux_input=*/nullptr,
    /*aux_input_to_input_weights=*/nullptr,
    /*aux_input_to_forget_weights=*/nullptr,
    /*aux_input_to_cell_weights=*/nullptr,
    /*aux_input_to_output_weights=*/nullptr, fw_input_gate_bias(), fw_forget_gate_bias(),
    fw_cell_gate_bias(), fw_output_gate_bias(),

    fw_projection_weights(), fw_projection_bias(), &lstm_params,
    /*forward_sequence=*/true, time_major,
    /*output_offset=*/0, fw__scratch_buffer, fw__output_state, fw__cell_state, fw_output());
  lstm::EvalFloat(
    input(), bw_input_to_input_weights(), bw_input_to_forget_weights(), bw_input_to_cell_weights(),
    bw_input_to_output_weights(),

    bw_recurrent_to_input_weights(), bw_recurrent_to_forget_weights(),
    bw_recurrent_to_cell_weights(), bw_recurrent_to_output_weights(),

    bw_cell_to_input_weights(), bw_cell_to_forget_weights(), bw_cell_to_output_weights(),

    t_bw_input_layer_norm_coefficients, t_bw_forget_layer_norm_coefficients,
    t_bw_cell_layer_norm_coefficients, t_bw_output_layer_norm_coefficients,
    /*aux_input=*/nullptr,
    /*aux_input_to_input_weights=*/nullptr,
    /*aux_input_to_forget_weights=*/nullptr,
    /*aux_input_to_cell_weights=*/nullptr,
    /*aux_input_to_output_weights=*/nullptr, bw_input_gate_bias(), bw_forget_gate_bias(),
    bw_cell_gate_bias(), bw_output_gate_bias(),

    bw_projection_weights(), bw_projection_bias(), &lstm_params,
    /*forward_sequence=*/true, time_major,
    /*output_offset=*/0, bw__scratch_buffer, bw__output_state, bw__cell_state, bw_output());
}

} // namespace kernels
} // namespace luci_interpreter
