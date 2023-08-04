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

#include "PALUnidirectionalSequenceLSTMCommon.h"

#include <arm_nnfunctions.h>

namespace luci_interpreter_pal
{
namespace
{

void precomputeZeroPointTimesWeightWithBias(int32_t zero_point, const circle::Tensor *weight_tensor,
                                            const circle::Tensor *bias_tensor,
                                            std::unique_ptr<int32_t> &output,
                                            luci_interpreter::BaseRuntimeGraph *runtime_graph)
{
  if (weight_tensor == nullptr)
    return;

  const int row = luci_interpreter::Tensor::dim(weight_tensor, 0);
  const int col = luci_interpreter::Tensor::dim(weight_tensor, 1);

  {
    auto output_ptr = new int32_t[row * sizeof(int32_t)];
    output.reset(output_ptr);
  }
  if (bias_tensor == nullptr)
  {
    std::memset(output.get(), 0, row * sizeof(int32_t));
  }
  else
  {
    const int32_t *bias = luci_interpreter::kernels::getTensorData<int32_t>(
      runtime_graph->getConstDataByTensor(bias_tensor));
    std::memcpy(output.get(), bias, row * sizeof(int32_t));
  }

  if (zero_point != 0)
  {
    const int8_t *weight = luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(weight_tensor));
    luci_interpreter::kernels::matrixScalarMultiplyAccumulate(weight, zero_point, row, col,
                                                              output.get());
  }
}

} // namespace

// Evaluate the LSTM kernel with (potential) multi-steps and multi-batch input
template <>
void evalLSTM<int8_t, int8_t, int16_t, int32_t>(
  luci_interpreter::lstm::LSTMStruct *lstm_struct,
  luci_interpreter::lstm::LSTMParameters *lstm_params,
  luci_interpreter::lstm::CellStateInfo *cell_state_info, int8_t *output_state_data,
  int16_t *cell_state_data, int16_t *scratch0, int16_t *scratch1, int16_t *scratch2,
  int16_t *scratch3, luci_interpreter::BaseRuntimeGraph *runtime_graph)
{
  // Pre-calculate bias + zero_point * weight.
  std::unique_ptr<int32_t> input_to_forget_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_forget_effective_bias;
  std::unique_ptr<int32_t> input_to_cell_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_cell_effective_bias;
  std::unique_ptr<int32_t> input_to_output_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_output_effective_bias;
  std::unique_ptr<int32_t> input_to_input_effective_bias;
  std::unique_ptr<int32_t> recurrent_to_input_effective_bias;

  const int32_t input_zero_point = -luci_interpreter::Tensor::zero_point(lstm_struct->input());
  const int32_t output_state_zero_point =
    -luci_interpreter::Tensor::zero_point(lstm_struct->output_state());

  precomputeZeroPointTimesWeightWithBias(input_zero_point, lstm_struct->input_to_forget_weights(),
                                         lstm_struct->forget_gate_bias(),
                                         input_to_forget_effective_bias, runtime_graph);

  precomputeZeroPointTimesWeightWithBias(output_state_zero_point,
                                         lstm_struct->recurrent_to_forget_weights(), nullptr,
                                         recurrent_to_forget_effective_bias, runtime_graph);

  precomputeZeroPointTimesWeightWithBias(input_zero_point, lstm_struct->input_to_cell_weights(),
                                         lstm_struct->cell_gate_bias(),
                                         input_to_cell_effective_bias, runtime_graph);

  precomputeZeroPointTimesWeightWithBias(output_state_zero_point,
                                         lstm_struct->recurrent_to_cell_weights(), nullptr,
                                         recurrent_to_cell_effective_bias, runtime_graph);

  precomputeZeroPointTimesWeightWithBias(input_zero_point, lstm_struct->input_to_output_weights(),
                                         lstm_struct->output_gate_bias(),
                                         input_to_output_effective_bias, runtime_graph);

  precomputeZeroPointTimesWeightWithBias(output_state_zero_point,
                                         lstm_struct->recurrent_to_output_weights(), nullptr,
                                         recurrent_to_output_effective_bias, runtime_graph);

  precomputeZeroPointTimesWeightWithBias(input_zero_point, lstm_struct->input_to_input_weights(),
                                         lstm_struct->input_gate_bias(),
                                         input_to_input_effective_bias, runtime_graph);

  precomputeZeroPointTimesWeightWithBias(output_state_zero_point,
                                         lstm_struct->recurrent_to_input_weights(), nullptr,
                                         recurrent_to_input_effective_bias, runtime_graph);

  cmsis_nn_lstm_params cmsis_lstm_params;
  cmsis_lstm_params.output_state_offset = -output_state_zero_point;
  cmsis_lstm_params.i2f_effective_bias = input_to_forget_effective_bias.get();
  cmsis_lstm_params.r2f_effective_bias = recurrent_to_forget_effective_bias.get();
  cmsis_lstm_params.i2c_effective_bias = input_to_cell_effective_bias.get();
  cmsis_lstm_params.r2c_effective_bias = recurrent_to_cell_effective_bias.get();
  cmsis_lstm_params.i2o_effective_bias = input_to_output_effective_bias.get();
  cmsis_lstm_params.r2o_effective_bias = recurrent_to_output_effective_bias.get();
  cmsis_lstm_params.i2i_effective_bias = input_to_input_effective_bias.get();
  cmsis_lstm_params.r2i_effective_bias = recurrent_to_input_effective_bias.get();

  // Get intermediate scales and zero points.
  float intermediate_scale[5];
  int32_t intermediate_zp[5];
  for (int i = 0; i < 4; ++i)
  {
    // Q3.12 for activation functions.
    intermediate_scale[i] = std::pow(2.0f, -12.0f);
    intermediate_zp[i] = 0;
  }

  const auto t = runtime_graph->findIntermediateTensor();
  assert(t != nullptr);
  const auto sc = luci_interpreter::Tensor::scale(t);
  const auto zer = luci_interpreter::Tensor::zero_point(t);

  intermediate_scale[4] = sc;
  intermediate_zp[4] = zer;

  // Scales.
  const float default_scale = 1.0;
  float input_scale = default_scale;
  float input_to_input_weight_scale = default_scale;
  float recurrent_to_input_weight_scale = default_scale;
  float input_to_forget_weight_scale = default_scale;
  float recurrent_to_forget_weight_scale = default_scale;
  float input_to_cell_weight_scale = default_scale;
  float recurrent_to_cell_weight_scale = default_scale;
  float input_to_output_weight_scale = default_scale;
  float recurrent_to_output_weight_scale = default_scale;
  float output_state_scale = default_scale;
  int cell_scale = 1;

  // Effective scales.
  float effective_input_to_input_scale = default_scale;
  float effective_recurrent_to_input_scale = default_scale;
  float effective_cell_to_input_scale = default_scale;
  float effective_input_to_forget_scale = default_scale;
  float effective_recurrent_to_forget_scale = default_scale;
  float effective_cell_to_forget_scale = default_scale;
  float effective_input_to_cell_scale = default_scale;
  float effective_recurrent_to_cell_scale = default_scale;
  float effective_input_to_output_scale = default_scale;
  float effective_recurrent_to_output_scale = default_scale;
  float effective_cell_to_output_scale = default_scale;
  float effective_hidden_scale = default_scale;

  // Populate scales.
  input_to_input_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->input_to_input_weights());
  recurrent_to_input_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->recurrent_to_input_weights());

  output_state_scale = luci_interpreter::Tensor::scale(lstm_struct->output_state());

  input_to_forget_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->input_to_forget_weights());
  input_to_cell_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->input_to_cell_weights());
  input_to_output_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->input_to_output_weights());
  recurrent_to_forget_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->recurrent_to_forget_weights());
  recurrent_to_cell_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->recurrent_to_cell_weights());
  recurrent_to_output_weight_scale =
    luci_interpreter::Tensor::scale(lstm_struct->recurrent_to_output_weights());

  luci_interpreter::kernels::checkedLog2(luci_interpreter::Tensor::scale(lstm_struct->cell_state()),
                                         &cell_scale);
  cmsis_lstm_params.cell_state_shift = cell_scale;
  input_scale = luci_interpreter::Tensor::scale(lstm_struct->input());

  // Calculate effective scales.
  effective_input_to_input_scale =
    input_to_input_weight_scale * input_scale / intermediate_scale[0];
  effective_recurrent_to_input_scale =
    recurrent_to_input_weight_scale * output_state_scale / intermediate_scale[0];

  effective_input_to_forget_scale =
    input_to_forget_weight_scale * input_scale / intermediate_scale[1];
  effective_recurrent_to_forget_scale =
    recurrent_to_forget_weight_scale * output_state_scale / intermediate_scale[1];

  effective_input_to_cell_scale = input_to_cell_weight_scale * input_scale / intermediate_scale[2];
  effective_recurrent_to_cell_scale =
    recurrent_to_cell_weight_scale * output_state_scale / intermediate_scale[2];

  effective_input_to_output_scale =
    input_to_output_weight_scale * input_scale / intermediate_scale[3];
  effective_recurrent_to_output_scale =
    recurrent_to_output_weight_scale * output_state_scale / intermediate_scale[3];

  effective_hidden_scale = std::pow(2.0f, -15.0f) / intermediate_scale[4] * std::pow(2.0f, -15.0f);

  // Decompose scales.
  int shift_output;
  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_input_to_input_scale),
    &cmsis_lstm_params.input_to_input_scaling.multiplier, &shift_output);
  cmsis_lstm_params.input_to_input_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_recurrent_to_input_scale),
    &cmsis_lstm_params.recurrent_to_input_scaling.multiplier, &shift_output);
  cmsis_lstm_params.recurrent_to_input_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(static_cast<double>(effective_cell_to_input_scale),
                                                &cmsis_lstm_params.cell_to_input_scaling.multiplier,
                                                &shift_output);
  cmsis_lstm_params.cell_to_input_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_input_to_forget_scale),
    &cmsis_lstm_params.input_to_forget_scaling.multiplier, &shift_output);
  cmsis_lstm_params.input_to_forget_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_recurrent_to_forget_scale),
    &cmsis_lstm_params.recurrent_to_forget_scaling.multiplier, &shift_output);
  cmsis_lstm_params.recurrent_to_forget_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_cell_to_forget_scale),
    &cmsis_lstm_params.cell_to_forget_scaling.multiplier, &shift_output);
  // ok
  cmsis_lstm_params.cell_to_forget_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(static_cast<double>(effective_input_to_cell_scale),
                                                &cmsis_lstm_params.input_to_cell_scaling.multiplier,
                                                &shift_output);
  cmsis_lstm_params.input_to_cell_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_recurrent_to_cell_scale),
    &cmsis_lstm_params.recurrent_to_cell_scaling.multiplier, &shift_output);
  cmsis_lstm_params.recurrent_to_cell_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_input_to_output_scale),
    &cmsis_lstm_params.input_to_output_scaling.multiplier, &shift_output);
  cmsis_lstm_params.input_to_output_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_recurrent_to_output_scale),
    &cmsis_lstm_params.recurrent_to_output_scaling.multiplier, &shift_output);
  cmsis_lstm_params.recurrent_to_output_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(
    static_cast<double>(effective_cell_to_output_scale),
    &cmsis_lstm_params.cell_to_output_scaling.multiplier, &shift_output);
  cmsis_lstm_params.cell_to_output_scaling.shift = static_cast<int32_t>(shift_output);

  cmsis_lstm_params.projection_scaling.shift = static_cast<int32_t>(shift_output);

  luci_interpreter::kernels::quantizeMultiplier(static_cast<double>(effective_hidden_scale),
                                                &cmsis_lstm_params.hidden_scaling.multiplier,
                                                &shift_output);
  cmsis_lstm_params.hidden_scaling.shift = static_cast<int32_t>(shift_output);

  cmsis_lstm_params.hidden_offset = intermediate_zp[4];

  cmsis_lstm_params.activation.min = std::numeric_limits<int16_t>::min();
  cmsis_lstm_params.activation.max = std::numeric_limits<int16_t>::max();

  cmsis_nn_lstm_context scratch_buffers;
  scratch_buffers.input_gate = scratch0;
  scratch_buffers.forget_gate = scratch1;
  scratch_buffers.cell_gate = scratch2;
  scratch_buffers.output_gate = scratch3;

  auto cell_clip = lstm_struct->options->cell_clip();

  if (cell_clip > 0.0f)
  {
    cmsis_lstm_params.clip.cell = static_cast<int16_t>(std::min(
      std::max(cell_clip / luci_interpreter::Tensor::scale(lstm_struct->cell_state()), -32768.0f),
      32767.0f));
  }
  else
  {
    cmsis_lstm_params.clip.cell = 0;
  }

  cmsis_lstm_params.clip.projection = 0;

  cmsis_lstm_params.time_major = lstm_struct->options->time_major();

  cmsis_lstm_params.input_gate_bias =
    const_cast<int32_t *>(luci_interpreter::kernels::getTensorData<int32_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->input_gate_bias())));
  cmsis_lstm_params.forget_gate_bias =
    const_cast<int32_t *>(luci_interpreter::kernels::getTensorData<int32_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->forget_gate_bias())));
  cmsis_lstm_params.cell_gate_bias =
    const_cast<int32_t *>(luci_interpreter::kernels::getTensorData<int32_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->cell_gate_bias())));
  cmsis_lstm_params.output_gate_bias =
    const_cast<int32_t *>(luci_interpreter::kernels::getTensorData<int32_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->output_gate_bias())));

  const auto input_dims_size = luci_interpreter::Tensor::num_dims(lstm_struct->input());
  const bool time_major = lstm_struct->options->time_major();
  const int n_input = luci_interpreter::Tensor::dim(lstm_struct->input(), input_dims_size - 1);
  const int n_output = luci_interpreter::Tensor::dim(lstm_struct->recurrent_to_output_weights(), 1);
  ;

  int max_time, n_batch;
  if (input_dims_size == 2)
  {
    max_time = 1;
    n_batch = luci_interpreter::Tensor::dim(lstm_struct->input(), 0);
  }
  else
  {
    max_time = (time_major) ? luci_interpreter::Tensor::dim(lstm_struct->input(), 0)
                            : luci_interpreter::Tensor::dim(lstm_struct->input(), 1);
    n_batch = (time_major) ? luci_interpreter::Tensor::dim(lstm_struct->input(), 1)
                           : luci_interpreter::Tensor::dim(lstm_struct->input(), 0);
  }

  cmsis_nn_lstm_dims lstm_dims;
  lstm_dims.num_inputs = n_input;
  lstm_dims.num_outputs = n_output;
  lstm_dims.num_batches = n_batch;
  lstm_dims.max_time = max_time;

  arm_lstm_unidirectional_s16_s8(
    &scratch_buffers,
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getDataByTensor(lstm_struct->input()))),
    &lstm_dims,
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->input_to_input_weights()))),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->input_to_forget_weights()))),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->input_to_cell_weights()))),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->input_to_output_weights()))),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_input_weights()))),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_forget_weights()))),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_cell_weights()))),
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->recurrent_to_output_weights()))),
    const_cast<int16_t *>(luci_interpreter::kernels::getTensorData<int16_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->cell_to_input_weights()))),
    const_cast<int16_t *>(luci_interpreter::kernels::getTensorData<int16_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->cell_to_forget_weights()))),
    const_cast<int16_t *>(luci_interpreter::kernels::getTensorData<int16_t>(
      runtime_graph->getConstDataByTensor(lstm_struct->cell_to_output_weights()))),
    nullptr, &cmsis_lstm_params, output_state_data, cell_state_data,
    const_cast<int8_t *>(luci_interpreter::kernels::getTensorData<int8_t>(
      runtime_graph->getDataByTensor(lstm_struct->output()))));
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
