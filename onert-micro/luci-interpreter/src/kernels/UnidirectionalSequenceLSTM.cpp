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
#include "kernels/UnidirectionalSequenceLSTM.h"
#include "kernels/Utils.h"

#include "PALUnidirectionalSequenceLSTM.h"

#include <cmath>

namespace luci_interpreter
{
namespace kernels
{
namespace
{
template <typename T> void fill_tensor_with_values(Tensor *state_tensor, T val)
{
  switch (state_tensor->element_type())
  {
    case DataType::FLOAT32:
    {
      auto state_data = state_tensor->data<float>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), val);
      break;
    }
    case DataType::S32:
    {
      auto state_data = state_tensor->data<int32_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), val);
      break;
    }
    case DataType::S16:
    {
      auto state_data = state_tensor->data<int16_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), val);
      break;
    }
    case DataType::S8:
    {
      auto state_data = state_tensor->data<int8_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), val);
      break;
    }
    case DataType::U8:
    {
      auto state_data = state_tensor->data<uint8_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), val);
      break;
    }
    default:
      assert(false && "Unsupported type.");
  }
}
} // namespace

UnidirectionalSequenceLSTM::UnidirectionalSequenceLSTM(
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

  const Tensor *input_layer_norm_coefficients, const Tensor *forget_layer_norm_coefficients,
  const Tensor *cell_layer_norm_coefficients, const Tensor *output_layer_norm_coefficients,

  std::vector<Tensor *> &&outputs, const UnidirectionalSequenceLSTMParams &params)
  : KernelWithParams<UnidirectionalSequenceLSTMParams>({input,
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

                                                        input_layer_norm_coefficients,
                                                        forget_layer_norm_coefficients,
                                                        cell_layer_norm_coefficients,
                                                        output_layer_norm_coefficients},
                                                       std::move(outputs), params)
{
  // Do nothing
}

// Check that input tensor dimensions matches with each other.
void UnidirectionalSequenceLSTM::check_input_tensor_dimensions(int n_input, int n_output,
                                                               int n_cell, bool use_layer_norm,
                                                               bool is_integer)
{
  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  LUCI_INTERPRETER_CHECK(params().cell_clip >= 0);
  LUCI_INTERPRETER_CHECK(params().proj_clip >= 0);

  if (input_to_input_weights() != nullptr)
  {
    const Shape &input_to_input_weights_shape = input_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.dim(1) == n_input);
  }

  const Shape &input_to_forget_weights_shape = input_to_forget_weights()->shape();
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.dim(1) == n_input);

  const Shape &input_to_cell_weights_shape = input_to_cell_weights()->shape();
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.dim(1) == n_input);

  if (recurrent_to_input_weights() != nullptr)
  {
    const Shape &recurrent_to_input_weights_shape = recurrent_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.dim(1) == n_output);
  }

  const Shape &recurrent_to_forget_weights_shape = recurrent_to_forget_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.dim(1) == n_output);

  const Shape &recurrent_to_cell_weights_shape = recurrent_to_cell_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.dim(1) == n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
    ((input_to_input_weights() != nullptr) && (recurrent_to_input_weights() != nullptr)) ||
    ((input_to_input_weights() == nullptr) && (recurrent_to_input_weights() == nullptr));
  LUCI_INTERPRETER_CHECK(cifg_weights_all_or_none == true);

  if (cell_to_input_weights() != nullptr)
  {
    const Shape &cell_to_input_weights_shape = cell_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_input_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_input_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer ? cell_to_input_weights()->element_type() == DataType::S16
                                      : cell_to_input_weights()->element_type() ==
                                          input_to_forget_weights()->element_type());
  }

  if (cell_to_forget_weights() != nullptr)
  {
    const Shape &cell_to_forget_weights_shape = cell_to_forget_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_forget_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_forget_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer ? cell_to_forget_weights()->element_type() == DataType::S16
                                      : cell_to_forget_weights()->element_type() ==
                                          input_to_forget_weights()->element_type());
  }

  if (cell_to_output_weights() != nullptr)
  {
    const Shape &cell_to_output_weights_shape = cell_to_output_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_output_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_output_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer ? cell_to_output_weights()->element_type() == DataType::S16
                                      : cell_to_output_weights()->element_type() ==
                                          input_to_forget_weights()->element_type());
  }

  // Making sure the peephole weights are there all or none.
  const bool use_cifg = (input_to_input_weights() == nullptr);
  const bool peephole_weights_all_or_none =
    ((cell_to_input_weights() != nullptr || use_cifg) && (cell_to_forget_weights() != nullptr) &&
     (cell_to_output_weights() != nullptr)) ||
    ((cell_to_input_weights() == nullptr) && (cell_to_forget_weights() == nullptr) &&
     (cell_to_output_weights() == nullptr));
  LUCI_INTERPRETER_CHECK(peephole_weights_all_or_none == true);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  if (use_cifg)
  {
    LUCI_INTERPRETER_CHECK(input_gate_bias() == nullptr);
  }
  else
  {
    const Shape &input_gate_bias_shape = input_gate_bias()->shape();
    LUCI_INTERPRETER_CHECK(input_gate_bias_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(input_gate_bias_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(input_gate_bias()->element_type() == DataType::S32);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(input_gate_bias()->element_type() == DataType::FLOAT32);
    }
  }

  const Shape &forget_gate_bias_shape = forget_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(forget_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(forget_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(forget_gate_bias()->element_type() == DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(forget_gate_bias()->element_type() == DataType::FLOAT32);
  }

  const Shape &cell_gate_bias_shape = cell_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(cell_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(cell_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(cell_gate_bias()->element_type() == DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(cell_gate_bias()->element_type() == DataType::FLOAT32);
  }

  const Shape &output_gate_bias_shape = output_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(output_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(output_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(output_gate_bias()->element_type() == DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(output_gate_bias()->element_type() == DataType::FLOAT32);
  }

  if (projection_weights() != nullptr)
  {
    const Shape &projection_weights_shape = projection_weights()->shape();
    LUCI_INTERPRETER_CHECK(projection_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(projection_weights_shape.dim(0) == n_output);
    LUCI_INTERPRETER_CHECK(projection_weights_shape.dim(1) == n_cell);
  }

  if (projection_bias() != nullptr)
  {
    const Shape &projection_bias_shape = projection_bias()->shape();
    LUCI_INTERPRETER_CHECK(projection_bias_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(projection_bias_shape.dim(0) == n_output);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(projection_bias()->element_type() == DataType::S32);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(projection_bias()->element_type() == DataType::FLOAT32);
    }
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  // TODO: make sure this is correct.
  const bool projecton_tensors_consistent =
    ((projection_weights() != nullptr) || (projection_bias() == nullptr));
  LUCI_INTERPRETER_CHECK(projecton_tensors_consistent == true);

  if (use_layer_norm)
  {
    if (use_cifg)
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients() == nullptr);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients() != nullptr)

      const Shape &input_layer_norm_coefficients_shape = input_layer_norm_coefficients()->shape();
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients_shape.num_dims() == 1);
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients_shape.dim(0) == n_cell);
      if (is_integer)
      {
        LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients()->element_type() == DataType::S16);
      }
      else
      {
        LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients()->element_type() ==
                               DataType::FLOAT32);
      }
    }

    const Shape &forget_layer_norm_coefficients_shape = forget_layer_norm_coefficients()->shape();
    LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients()->element_type() == DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients()->element_type() == DataType::FLOAT32);
    }

    const Shape &cell_layer_norm_coefficients_shape = cell_layer_norm_coefficients()->shape();
    LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients()->element_type() == DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients()->element_type() == DataType::FLOAT32);
    }

    const Shape &output_layer_norm_coefficients_shape = output_layer_norm_coefficients()->shape();
    LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients()->element_type() == DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients()->element_type() == DataType::FLOAT32);
    }
  }
}

void UnidirectionalSequenceLSTM::precompute_zero_point_times_weight_with_bias(
  int32_t zero_point, const Tensor *weight_tensor, const Tensor *bias_tensor,
  std::vector<int32_t> &output)
{
  if (weight_tensor == nullptr)
    return;

  const auto weight_shape = weight_tensor->shape();

  LUCI_INTERPRETER_CHECK(weight_shape.num_dims() == 2);
  const int row = weight_shape.dim(0);
  const int col = weight_shape.dim(1);

  output.resize(row);

  if (bias_tensor == nullptr)
  {
    memset(output.data(), 0, row * sizeof(int32_t));
  }
  else
  {
    const int32_t *bias = getTensorData<int32_t>(bias_tensor);
    std::memcpy(output.data(), bias, row * sizeof(int32_t));
  }
  if (zero_point != 0)
  {
    const int8_t *weight = getTensorData<int8_t>(weight_tensor);
    matrixScalarMultiplyAccumulate(weight, zero_point, row, col, output.data());
  }
}

void UnidirectionalSequenceLSTM::populate_precomputed_zp_times_weight_with_bias()
{
  const int32_t input_zero_point = -input()->zero_point();
  const int32_t output_state_zero_point = -output_state()->zero_point();

  const int32_t hidden_zp = params().intermediate_affine_quant[0]->zero_point[0];

  // Get bias and perform zero point calculation.
  // When there is layer normalization, the gate bias does not apply to matmul
  // directly:
  //      y = ln(w * x + w * r + w * c) + b.

  bool is_layer_norm = (forget_layer_norm_coefficients() != nullptr);

  const auto forget_gate_bias_tmp = is_layer_norm ? nullptr : forget_gate_bias();

  precompute_zero_point_times_weight_with_bias(input_zero_point, input_to_forget_weights(),
                                               forget_gate_bias_tmp,
                                               integer_lstm_params.input_to_forget_effective_bias);

  precompute_zero_point_times_weight_with_bias(
    output_state_zero_point, recurrent_to_forget_weights(), nullptr,
    integer_lstm_params.recurrent_to_forget_effective_bias);

  // Modulation gate.
  const auto cell_gate_bias_tmp = is_layer_norm ? nullptr : cell_gate_bias();

  precompute_zero_point_times_weight_with_bias(input_zero_point, input_to_cell_weights(),
                                               cell_gate_bias_tmp,
                                               integer_lstm_params.input_to_cell_effective_bias);
  precompute_zero_point_times_weight_with_bias(
    output_state_zero_point, recurrent_to_cell_weights(), nullptr,
    integer_lstm_params.recurrent_to_cell_effective_bias);

  // Output gate.
  const auto output_gate_bias_tmp = is_layer_norm ? nullptr : output_gate_bias();

  precompute_zero_point_times_weight_with_bias(input_zero_point, input_to_output_weights(),
                                               output_gate_bias_tmp,
                                               integer_lstm_params.input_to_output_effective_bias);
  precompute_zero_point_times_weight_with_bias(
    output_state_zero_point, recurrent_to_output_weights(), nullptr,
    integer_lstm_params.recurrent_to_output_effective_bias);

  // Input gate. The calculation is only meaningful for non-cifg case.
  const auto input_gate_bias_tmp = is_layer_norm ? nullptr : input_gate_bias();

  precompute_zero_point_times_weight_with_bias(input_zero_point, input_to_input_weights(),
                                               input_gate_bias_tmp,
                                               integer_lstm_params.input_to_input_effective_bias);
  precompute_zero_point_times_weight_with_bias(
    output_state_zero_point, recurrent_to_input_weights(), nullptr,
    integer_lstm_params.recurrent_to_input_effective_bias);

  // Projection bias. The calculation is only meaningful for with projection.
  precompute_zero_point_times_weight_with_bias(hidden_zp, projection_weights(), projection_bias(),
                                               integer_lstm_params.projection_effective_bias);
}

void UnidirectionalSequenceLSTM::populate_quantized_lstm_params()
{
  const float cell_clip = params().cell_clip;
  const float proj_clip = params().proj_clip;

  LUCI_INTERPRETER_CHECK(cell_state() != nullptr);
  LUCI_INTERPRETER_CHECK(cell_state()->scales().size() > 0 and
                         cell_state()->zero_points().size() > 0);

  LUCI_INTERPRETER_CHECK(output()->scales().size() > 0 and output()->zero_points().size() > 0);

  if (cell_clip > 0.0f)
  {
    integer_lstm_params.quantized_cell_clip = static_cast<int16_t>(
      std::min(std::max(cell_clip / cell_state()->scales()[0], -32768.0f), 32767.0f));
  }
  else
  {
    integer_lstm_params.quantized_cell_clip = 0;
  }

  if (proj_clip > 0.0f)
  {
    integer_lstm_params.quantized_proj_clip =
      static_cast<int8_t>(std::min(std::max(proj_clip / output()->scales()[0], -128.0f), 127.0f));
  }
  else
  {
    integer_lstm_params.quantized_proj_clip = 0;
  }

  // Calculate effective scales.
  bool use_layer_norm = (forget_layer_norm_coefficients() != nullptr);

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const bool use_cifg = (input_to_input_weights() == nullptr);
  const bool use_peephole = (cell_to_output_weights() != nullptr);
  const bool use_projection = (projection_weights() != nullptr);

  // Get intermediate scales and zero points.
  float intermediate_scale[5];
  int32_t intermediate_zp[5];

  for (int i = 0; i < 4; ++i)
  {
    if (use_layer_norm)
    {
      assert(false);
      // TODO: support it
    }
    else
    {
      intermediate_scale[i] = std::pow(2.0f, -12.0f);
      intermediate_zp[i] = 0;
    }
  }
  intermediate_scale[4] = params().intermediate_affine_quant[0]->scale[0];
  intermediate_zp[4] = params().intermediate_affine_quant[0]->zero_point[0];

  // Scales.
  const float default_scale = 1.0;
  float input_scale = default_scale;
  float input_to_input_weight_scale = default_scale;
  float recurrent_to_input_weight_scale = default_scale;
  float cell_to_input_weight_scale = default_scale;
  float input_to_forget_weight_scale = default_scale;
  float recurrent_to_forget_weight_scale = default_scale;
  float cell_to_forget_weight_scale = default_scale;
  float input_to_cell_weight_scale = default_scale;
  float recurrent_to_cell_weight_scale = default_scale;
  float input_to_output_weight_scale = default_scale;
  float recurrent_to_output_weight_scale = default_scale;
  float cell_to_output_weight_scale = default_scale;
  float projection_weight_scale = default_scale;
  float layer_norm_input_scale = default_scale;
  float layer_norm_forget_scale = default_scale;
  float layer_norm_cell_scale = default_scale;
  float layer_norm_output_scale = default_scale;
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
  float effective_proj_scale = default_scale;
  float effective_hidden_scale = default_scale;

  // Populate scales.
  if (!use_cifg)
  {
    input_to_input_weight_scale = input_to_input_weights()->scale();
    recurrent_to_input_weight_scale = recurrent_to_input_weights()->scale();
  }

  if (use_peephole)
  {
    if (!use_cifg)
    {
      cell_to_input_weight_scale = cell_to_input_weights()->scale();
    }
    cell_to_forget_weight_scale = cell_to_forget_weights()->scale();
    cell_to_output_weight_scale = cell_to_output_weights()->scale();
  }

  if (use_layer_norm)
  {
    if (!use_cifg)
    {
      layer_norm_input_scale = input_layer_norm_coefficients()->scale();
    }
    layer_norm_forget_scale = forget_layer_norm_coefficients()->scale();
    layer_norm_cell_scale = cell_layer_norm_coefficients()->scale();
    layer_norm_output_scale = output_layer_norm_coefficients()->scale();
  }

  if (use_projection)
  {
    projection_weight_scale = projection_weights()->scale();
  }
  output_state_scale = output_state()->scale();

  input_to_forget_weight_scale = input_to_forget_weights()->scale();
  input_to_cell_weight_scale = input_to_cell_weights()->scale();
  input_to_output_weight_scale = input_to_output_weights()->scale();
  recurrent_to_forget_weight_scale = recurrent_to_forget_weights()->scale();
  recurrent_to_cell_weight_scale = recurrent_to_cell_weights()->scale();
  recurrent_to_output_weight_scale = recurrent_to_output_weights()->scale();

  // Check cell state (already used above)
  {
    const float x_log2 = std::log(cell_state()->scale()) * (1.0f / std::log(2.0f));
    const float x_log2_rounded = std::round(x_log2);
    const float x_log2_fracpart = x_log2 - x_log2_rounded;
    cell_scale = static_cast<int>(x_log2_rounded);

    LUCI_INTERPRETER_CHECK(std::abs(x_log2_fracpart) < 1e-3f);
  }
  integer_lstm_params.cell_scale = cell_scale;
  input_scale = input()->scale();

  // Calculate effective scales.
  if (!use_cifg)
  {
    effective_input_to_input_scale =
      input_to_input_weight_scale * input_scale / intermediate_scale[0];
    effective_recurrent_to_input_scale =
      recurrent_to_input_weight_scale * output_state_scale / intermediate_scale[0];
  }
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

  effective_proj_scale = projection_weight_scale * intermediate_scale[4] / output_state_scale;

  if (use_peephole)
  {
    if (!use_cifg)
    {
      effective_cell_to_input_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
                                      cell_to_input_weight_scale / intermediate_scale[0];
    }
    effective_cell_to_forget_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
                                     cell_to_forget_weight_scale / intermediate_scale[1];
    effective_cell_to_output_scale = std::pow(2.0f, static_cast<float>(cell_scale)) *
                                     cell_to_output_weight_scale / intermediate_scale[3];
  }

  // Decompose scales.
  int shift_output = 0;

  quantizeMultiplier(static_cast<double>(effective_input_to_input_scale),
                     &integer_lstm_params.effective_input_to_input_scale_a, &shift_output);
  integer_lstm_params.effective_input_to_input_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_recurrent_to_input_scale),
                     &integer_lstm_params.effective_recurrent_to_input_scale_a, &shift_output);
  integer_lstm_params.effective_recurrent_to_input_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_cell_to_input_scale),
                     &integer_lstm_params.effective_cell_to_input_scale_a, &shift_output);
  integer_lstm_params.effective_cell_to_input_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_input_to_forget_scale),
                     &integer_lstm_params.effective_input_to_forget_scale_a, &shift_output);
  integer_lstm_params.effective_input_to_forget_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_recurrent_to_forget_scale),
                     &integer_lstm_params.effective_recurrent_to_forget_scale_a, &shift_output);
  integer_lstm_params.effective_recurrent_to_forget_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_cell_to_forget_scale),
                     &integer_lstm_params.effective_cell_to_forget_scale_a, &shift_output);
  integer_lstm_params.effective_cell_to_forget_scale_b = static_cast<int32_t>(shift_output);
  quantizeMultiplier(static_cast<double>(effective_input_to_cell_scale),
                     &integer_lstm_params.effective_input_to_cell_scale_a, &shift_output);
  integer_lstm_params.effective_input_to_cell_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_recurrent_to_cell_scale),
                     &integer_lstm_params.effective_recurrent_to_cell_scale_a, &shift_output);
  integer_lstm_params.effective_recurrent_to_cell_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_input_to_output_scale),
                     &integer_lstm_params.effective_input_to_output_scale_a, &shift_output);
  integer_lstm_params.effective_input_to_output_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_recurrent_to_output_scale),
                     &integer_lstm_params.effective_recurrent_to_output_scale_a, &shift_output);
  integer_lstm_params.effective_recurrent_to_output_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_cell_to_output_scale),
                     &integer_lstm_params.effective_cell_to_output_scale_a, &shift_output);
  integer_lstm_params.effective_cell_to_output_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_proj_scale),
                     &integer_lstm_params.effective_proj_scale_a, &shift_output);
  integer_lstm_params.effective_proj_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(effective_hidden_scale),
                     &integer_lstm_params.effective_hidden_scale_a, &shift_output);

  integer_lstm_params.effective_hidden_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(layer_norm_input_scale),
                     &integer_lstm_params.layer_norm_input_scale_a, &shift_output);
  integer_lstm_params.layer_norm_input_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(layer_norm_forget_scale),
                     &integer_lstm_params.layer_norm_forget_scale_a, &shift_output);
  integer_lstm_params.layer_norm_forget_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(layer_norm_cell_scale),
                     &integer_lstm_params.layer_norm_cell_scale_a, &shift_output);
  integer_lstm_params.layer_norm_cell_scale_b = static_cast<int32_t>(shift_output);

  quantizeMultiplier(static_cast<double>(layer_norm_output_scale),
                     &integer_lstm_params.layer_norm_output_scale_a, &shift_output);
  integer_lstm_params.layer_norm_output_scale_b = static_cast<int32_t>(shift_output);

  integer_lstm_params.hidden_zp = intermediate_zp[4];

  // 10000 is used to make sure the kernel logic does not overflow.
  if (!use_cifg)
  {
    integer_lstm_params.input_variance_guard =
      std::max(1, static_cast<int>(10000 * layer_norm_input_scale));
  }
  integer_lstm_params.forget_variance_guard =
    std::max(1, static_cast<int>(10000 * layer_norm_forget_scale));
  integer_lstm_params.cell_variance_guard =
    std::max(1, static_cast<int>(10000 * layer_norm_cell_scale));
  integer_lstm_params.output_variance_guard =
    std::max(1, static_cast<int>(10000 * layer_norm_output_scale));
}

void UnidirectionalSequenceLSTM::configure()
{
  LUCI_INTERPRETER_CHECK(getInputTensors().size() == 24 - 2);
  LUCI_INTERPRETER_CHECK(getOutputTensors().size() >= 1);

  bool use_layer_norm = (forget_layer_norm_coefficients() != nullptr);

  // TODO: check can we don't create forget_layer_norm_coefficients tensor
  // TODO: support S32
  LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::S8);

  const bool is_integer = input()->element_type() == DataType::S8;

  // TODO: support it
  if (use_layer_norm and is_integer)
    assert(false && "Not supported now");

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  const Shape &input_shape = input()->shape();
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() > 1);
  const bool time_major = params().time_major;
  const int n_batch = time_major ? input_shape.dim(1) : input_shape.dim(0);
  // NOTE as dim(2) is accessed, we need to check this is valid
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() > 2);
  const int n_input = input_shape.dim(2);

  const Shape &input_to_output_weights_shape = input_to_output_weights()->shape();
  const int n_cell = input_to_output_weights_shape.dim(0);
  LUCI_INTERPRETER_CHECK(input_to_output_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_output_weights_shape.dim(1) == n_input);

  const Shape &recurrent_to_output_weights_shape = recurrent_to_output_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_output_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_output_weights_shape.dim(0) == n_cell);

  const int n_output = recurrent_to_output_weights_shape.dim(1);

  // Check that input tensor dimensions matches with each other.
  check_input_tensor_dimensions(n_input, n_output, n_cell, use_layer_norm, is_integer);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  const Shape &output_state_shape = output_state()->shape();
  const Shape &cell_state_shape = cell_state()->shape();
  LUCI_INTERPRETER_CHECK(output_state_shape.num_elements() == n_batch * n_output);
  LUCI_INTERPRETER_CHECK(cell_state_shape.num_elements() == n_batch * n_cell);

  // Resize the output tensors.
  Shape output_shape = Shape(input_shape.num_dims());
  for (int i = 0; i < input_shape.num_dims() - 1; i++)
  {
    output_shape.dim(i) = input_shape.dim(i);
  }
  output_shape.dim(input_shape.num_dims() - 1) = n_output;
  output()->resize(output_shape);

  // Let's create temp tensors for integer LSTM
  const bool use_cifg = (input_to_input_weights() == nullptr);

  // TODO support float
  if (is_integer)
  {
    // Integer UnidirectionalSequenceLSTM prepare function for 8x8->16.
    // This code path needs 5 intermediate tensors per Op.
    // Populate quantization parameters.
    populate_quantized_lstm_params();

    LUCI_INTERPRETER_CHECK(_outputs.size() == 6 + 2);

    for (int i = 1; i < 6; ++i)
    {
      getOutputTensors().at(i)->resize({n_batch * n_cell});
    }

    populate_precomputed_zp_times_weight_with_bias();
  }
  else
  {
    assert(false && "Not impl yet, only integer");
  }
}

void UnidirectionalSequenceLSTM::execute() const
{
  switch (input_to_output_weights()->element_type())
  {
    case DataType::S8:
      evalInt8();
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

void UnidirectionalSequenceLSTM::evalInt8() const
{
  const bool use_layer_norm = (forget_layer_norm_coefficients() != nullptr);
  const bool time_major = params().time_major;
  const auto output_state_zero_point = output_state()->zero_point();

  fill_tensor_with_values(getOutputTensors()[1], 0);
  fill_tensor_with_values(getOutputTensors()[2], 0);
  fill_tensor_with_values(getOutputTensors()[3], 0);
  fill_tensor_with_values(getOutputTensors()[4], 0);
  fill_tensor_with_values(getOutputTensors()[5], 0);
  fill_tensor_with_values(output_state(), 0);
  fill_tensor_with_values(cell_state(), 0);

  luci_interpreter_pal::eval_integer_8x8_16_lstm(
    input(), input_to_input_weights(), input_to_forget_weights(), input_to_cell_weights(),
    input_to_output_weights(), recurrent_to_input_weights(), recurrent_to_forget_weights(),
    recurrent_to_cell_weights(), recurrent_to_output_weights(), cell_to_input_weights(),
    cell_to_forget_weights(), cell_to_output_weights(), input_layer_norm_coefficients(),
    forget_layer_norm_coefficients(), cell_layer_norm_coefficients(),
    output_layer_norm_coefficients(), input_gate_bias(), forget_gate_bias(), cell_gate_bias(),
    output_gate_bias(), projection_weights(), projection_bias(), params(),
    /*forward_sequence=*/true, time_major, integer_lstm_params, output_state_zero_point,
    output_state(), cell_state(), output(), getTensorData<int16_t>(getOutputTensors()[1]),
    getTensorData<int16_t>(getOutputTensors()[2]), getTensorData<int16_t>(getOutputTensors()[3]),
    getTensorData<int16_t>(getOutputTensors()[4]), getTensorData<int8_t>(getOutputTensors()[5]),
    nullptr);
}

} // namespace kernels
} // namespace luci_interpreter
