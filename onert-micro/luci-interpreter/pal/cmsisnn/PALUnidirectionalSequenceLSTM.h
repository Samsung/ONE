/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
#define LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H

#include "arm_nnfunctions.h"
#include "PALUnidirectionalSequenceLSTMCommon.h"

namespace luci_interpreter_pal
{
namespace lstm
{

inline cmsis_nn_lstm_params
convert_lstm_params(const luci_interpreter::IntegerLSTMParams &params_in, bool time_major,
                    int32_t output_zeropoint, const int32_t *input_gate_bias,
                    const int32_t *forget_gate_bias, const int32_t *cell_gate_bias,
                    const int32_t *output_gate_bias, int16_t *input_layer_norm_coefficients,
                    int16_t *forget_layer_norm_coefficients, int16_t *cell_layer_norm_coefficients,
                    int16_t *output_layer_norm_coefficients)
{
  cmsis_nn_lstm_params params_out;

  params_out.time_major = time_major;

  // Multipliers and shifts for weights
  params_out.input_to_input_scaling.multiplier = params_in.effective_input_to_input_scale_a;
  params_out.input_to_input_scaling.shift = params_in.effective_input_to_input_scale_b;
  params_out.recurrent_to_input_scaling.multiplier = params_in.effective_recurrent_to_input_scale_a;
  params_out.recurrent_to_input_scaling.shift = params_in.effective_recurrent_to_input_scale_b;
  params_out.cell_to_input_scaling.multiplier = params_in.effective_cell_to_input_scale_a;
  params_out.cell_to_input_scaling.shift = params_in.effective_cell_to_input_scale_b;
  params_out.input_to_forget_scaling.multiplier = params_in.effective_input_to_forget_scale_a;
  params_out.input_to_forget_scaling.shift = params_in.effective_input_to_forget_scale_b;
  params_out.recurrent_to_forget_scaling.multiplier =
    params_in.effective_recurrent_to_forget_scale_a;
  params_out.recurrent_to_forget_scaling.shift = params_in.effective_recurrent_to_forget_scale_b;
  params_out.cell_to_forget_scaling.multiplier = params_in.effective_cell_to_forget_scale_a;
  params_out.cell_to_forget_scaling.shift = params_in.effective_cell_to_forget_scale_b;
  params_out.input_to_cell_scaling.multiplier = params_in.effective_input_to_cell_scale_a;
  params_out.input_to_cell_scaling.shift = params_in.effective_input_to_cell_scale_b;
  params_out.recurrent_to_cell_scaling.multiplier = params_in.effective_recurrent_to_cell_scale_a;
  params_out.recurrent_to_cell_scaling.shift = params_in.effective_recurrent_to_cell_scale_b;
  params_out.input_to_output_scaling.multiplier = params_in.effective_input_to_output_scale_a;
  params_out.input_to_output_scaling.shift = params_in.effective_input_to_output_scale_b;

  params_out.recurrent_to_output_scaling.multiplier =
    params_in.effective_recurrent_to_output_scale_a;
  params_out.recurrent_to_output_scaling.shift = params_in.effective_recurrent_to_output_scale_b;
  params_out.cell_to_output_scaling.multiplier = params_in.effective_cell_to_output_scale_a;
  params_out.cell_to_output_scaling.shift = params_in.effective_cell_to_output_scale_b;
  params_out.projection_scaling.multiplier = params_in.effective_proj_scale_a;
  params_out.projection_scaling.shift = params_in.effective_proj_scale_b;

  params_out.layer_norm_input_scaling.multiplier = params_in.layer_norm_input_scale_a;
  params_out.layer_norm_input_scaling.shift = params_in.layer_norm_input_scale_b;
  params_out.layer_norm_forget_scaling.multiplier = params_in.layer_norm_forget_scale_a;
  params_out.layer_norm_forget_scaling.shift = params_in.layer_norm_forget_scale_b;
  params_out.layer_norm_cell_scaling.multiplier = params_in.layer_norm_cell_scale_a;
  params_out.layer_norm_cell_scaling.shift = params_in.layer_norm_cell_scale_b;
  params_out.layer_norm_output_scaling.multiplier = params_in.layer_norm_output_scale_a;
  params_out.layer_norm_output_scaling.shift = params_in.layer_norm_output_scale_b;

  params_out.clip.cell = params_in.quantized_cell_clip;
  params_out.clip.projection = params_in.quantized_proj_clip;

  params_out.cell_state_shift = params_in.cell_scale;

  params_out.hidden_offset = params_in.hidden_zp;
  params_out.output_state_offset = output_zeropoint;

  params_out.guard.input_variance = params_in.input_variance_guard;
  params_out.guard.forget_variance = params_in.forget_variance_guard;
  params_out.guard.cell_variance = params_in.cell_variance_guard;
  params_out.guard.output_variance = params_in.output_variance_guard;

  params_out.i2f_effective_bias = params_in.input_to_forget_effective_bias.data();
  params_out.r2f_effective_bias = params_in.recurrent_to_forget_effective_bias.data();
  params_out.i2c_effective_bias = params_in.input_to_cell_effective_bias.data();
  params_out.r2c_effective_bias = params_in.recurrent_to_cell_effective_bias.data();
  params_out.i2o_effective_bias = params_in.input_to_output_effective_bias.data();
  params_out.r2o_effective_bias = params_in.recurrent_to_output_effective_bias.data();
  params_out.i2i_effective_bias = params_in.input_to_input_effective_bias.data();
  params_out.r2i_effective_bias = params_in.recurrent_to_input_effective_bias.data();
  params_out.projection_effective_bias = params_in.projection_effective_bias.data();

  params_out.hidden_scaling.multiplier = params_in.effective_hidden_scale_a;
  params_out.hidden_scaling.shift = params_in.effective_hidden_scale_b;

  params_out.input_gate_bias = input_gate_bias;
  params_out.forget_gate_bias = forget_gate_bias;
  params_out.cell_gate_bias = cell_gate_bias;
  params_out.output_gate_bias = output_gate_bias;

  params_out.layer_norm.input_weight = input_layer_norm_coefficients;
  params_out.layer_norm.forget_weight = forget_layer_norm_coefficients;
  params_out.layer_norm.cell_weight = cell_layer_norm_coefficients;
  params_out.layer_norm.output_weight = output_layer_norm_coefficients;

  params_out.activation.min = std::numeric_limits<int16_t>::min();
  params_out.activation.max = std::numeric_limits<int16_t>::max();

  return params_out;
}

} // namespace lstm

void eval_integer_8x8_16_lstm(
  const luci_interpreter::Tensor *input, const luci_interpreter::Tensor *input_to_input_weights,
  const luci_interpreter::Tensor *input_to_forget_weights,
  const luci_interpreter::Tensor *input_to_cell_weights,
  const luci_interpreter::Tensor *input_to_output_weights,
  const luci_interpreter::Tensor *recurrent_to_input_weights,
  const luci_interpreter::Tensor *recurrent_to_forget_weights,
  const luci_interpreter::Tensor *recurrent_to_cell_weights,
  const luci_interpreter::Tensor *recurrent_to_output_weights,
  const luci_interpreter::Tensor *cell_to_input_weights,
  const luci_interpreter::Tensor *cell_to_forget_weights,
  const luci_interpreter::Tensor *cell_to_output_weights,
  const luci_interpreter::Tensor *input_layer_norm_coefficients,
  const luci_interpreter::Tensor *forget_layer_norm_coefficients,
  const luci_interpreter::Tensor *cell_layer_norm_coefficients,
  const luci_interpreter::Tensor *output_layer_norm_coefficients,
  const luci_interpreter::Tensor *input_gate_bias, const luci_interpreter::Tensor *forget_gate_bias,
  const luci_interpreter::Tensor *cell_gate_bias, const luci_interpreter::Tensor *output_gate_bias,
  const luci_interpreter::Tensor *projection_weights,
  const luci_interpreter::Tensor *projection_bias,
  const luci_interpreter::UnidirectionalSequenceLSTMParams &params, bool forward_sequence,
  bool time_major, const luci_interpreter::IntegerLSTMParams &integer_lstm_param,
  int32_t output_state_zp, luci_interpreter::Tensor *output_state,
  luci_interpreter::Tensor *cell_state, luci_interpreter::Tensor *output, int16_t *scratch0,
  int16_t *scratch1, int16_t *scratch2, int16_t *scratch3, int8_t *scratch4, int32_t *scratch5)
{
  // CMSIS-NN does not support these configurations currently.
  // Please use MCU kernels instead
  const bool use_layer_norm = (forget_layer_norm_coefficients != nullptr);
  const bool use_peephole = (cell_to_output_weights != nullptr);
  const bool use_projection = (projection_weights != nullptr);
  const bool use_cifg = (input_to_input_weights == nullptr);
  const bool unsupported_config = use_layer_norm || use_peephole || use_projection || use_cifg;

  if (unsupported_config)
  {
    assert(false && "CMSIS-NN does not support these configurations currently");
    return;
  }

  const auto input_shape = input->shape();
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() >= 2 && input_shape.num_dims() <= 3);

  cmsis_nn_lstm_context scratch_buffers;
  scratch_buffers.input_gate = scratch0;
  scratch_buffers.forget_gate = scratch1;
  scratch_buffers.cell_gate = scratch2;
  scratch_buffers.output_gate = scratch3;
  scratch_buffers.scratch = scratch4;

  cmsis_nn_lstm_params cmsis_lstm_params = lstm::convert_lstm_params(
    integer_lstm_param, time_major, output_state_zp,
    luci_interpreter::kernels::getTensorData<int32_t>(input_gate_bias),
    luci_interpreter::kernels::getTensorData<int32_t>(forget_gate_bias),
    luci_interpreter::kernels::getTensorData<int32_t>(cell_gate_bias),
    luci_interpreter::kernels::getTensorData<int32_t>(output_gate_bias),
    const_cast<int16_t *>(
      luci_interpreter::kernels::getTensorData<int16_t>(input_layer_norm_coefficients)),
    const_cast<int16_t *>(
      luci_interpreter::kernels::getTensorData<int16_t>(forget_layer_norm_coefficients)),
    const_cast<int16_t *>(
      luci_interpreter::kernels::getTensorData<int16_t>(cell_layer_norm_coefficients)),
    const_cast<int16_t *>(
      luci_interpreter::kernels::getTensorData<int16_t>(output_layer_norm_coefficients)));

  const int n_input = input_shape.dim(input_shape.num_dims() - 1);
  int max_time, n_batch;
  if (input_shape.num_dims() == 2)
  {
    max_time = 1;
    n_batch = input_shape.dim(0);
  }
  else
  {
    max_time = (time_major) ? input_shape.dim(0) : input_shape.dim(1);
    n_batch = (time_major) ? input_shape.dim(1) : input_shape.dim(0);
  }

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->shape().dim(0);
  const int n_output = recurrent_to_output_weights->shape().dim(1);

  cmsis_nn_lstm_dims lstm_dims;
  lstm_dims.num_inputs = n_input;
  lstm_dims.num_outputs = n_output;
  lstm_dims.num_batches = n_batch;
  lstm_dims.max_time = max_time;

  arm_lstm_unidirectional_s16_s8(
    &scratch_buffers, const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(input)),
    &lstm_dims,
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(input_to_input_weights)),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(input_to_forget_weights)),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(input_to_cell_weights)),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(input_to_output_weights)),
    const_cast<int8_t *>(
      luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_input_weights)),
    const_cast<int8_t *>(
      luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_forget_weights)),
    const_cast<int8_t *>(
      luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_cell_weights)),
    const_cast<int8_t *>(
      luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_output_weights)),
    const_cast<int16_t *>(luci_interpreter::kernels::getTensorData<int16_t>(cell_to_input_weights)),
    const_cast<int16_t *>(
      luci_interpreter::kernels::getTensorData<int16_t>(cell_to_forget_weights)),
    const_cast<int16_t *>(
      luci_interpreter::kernels::getTensorData<int16_t>(cell_to_output_weights)),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(projection_weights)),
    &cmsis_lstm_params,
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(output_state)),
    const_cast<int16_t *>(luci_interpreter::kernels::getTensorData<int16_t>(cell_state)),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(output)));
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
