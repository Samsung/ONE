/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_EXECUTE_PAL_GRU_COMMON_H
#define ONERT_MICRO_EXECUTE_PAL_GRU_COMMON_H

#include "OMStatus.h"
#include "core/OMRuntimeShape.h"

#include "PALUtils.h"
#include "ProcessBroadcastShapes.h"
#include "PALFullyConnected.h"
#include "PALLogistic.h"

namespace onert_micro
{
namespace execute
{
namespace pal
{
namespace
{
void calculateGRU(const float *input_data, const float *weight_input_data,
                  const float *weight_hidden_data, const float *bias_input_data,
                  const float *bias_hidden_data, float *output_data,
                  const core::OMRuntimeShape &input_shape, const core::OMRuntimeShape &output_shape,
                  const core::OMRuntimeShape &weight_input_shape,
                  const core::OMRuntimeShape &weight_hidden_shape, float *output_input_data,
                  float *output_hidden_data, const core::OMRuntimeShape &output_shape_fc)
{
  core::FullyConnectedParams op_params{};
  // As FC nodes doesn't have any activations inside GRU, let' use just numeric limits
  op_params.float_activation_min = std::numeric_limits<float>::lowest();
  op_params.float_activation_max = std::numeric_limits<float>::max();

  // FC Input
  FullyConnected(op_params, output_data, weight_input_shape, weight_input_data, bias_input_data,
                 output_shape_fc, output_input_data);

  // FC Hidden
  FullyConnected(op_params, input_data, weight_hidden_shape, weight_hidden_data, bias_hidden_data,
                 output_shape_fc, output_hidden_data);

  int num_elements = output_shape_fc.dims(1) / 3;

  float *second_hidden_part = output_hidden_data + num_elements;
  float *second_input_part = output_input_data + num_elements;

  float *third_hidden_part = second_hidden_part + num_elements;
  float *third_input_part = second_input_part + num_elements;

  // Calculate Left part
  for (int i = 0; i < num_elements; ++i)
  {
    output_input_data[i] += output_hidden_data[i];
  }

  Logistic(num_elements, output_input_data, output_input_data);

  // Calculate most left add
  float *most_left_part_final = output_input_data;
  float *first_part = output_input_data;
  for (int i = 0; i < num_elements; ++i)
  {
    output_data[i] *= most_left_part_final[i];
    first_part[i] = 1.0f - first_part[i];
  }

  // Calc third part
  for (int i = 0; i < num_elements; ++i)
  {
    second_hidden_part[i] += second_input_part[i];
  }
  Logistic(num_elements, second_hidden_part, second_hidden_part);

  for (int i = 0; i < num_elements; ++i)
  {
    second_hidden_part[i] *= third_input_part[i];
    second_hidden_part[i] += third_hidden_part[i];
    second_hidden_part[i] = std::tanh(second_hidden_part[i]);
    second_hidden_part[i] *= first_part[i];
    output_data[i] += second_hidden_part[i];
  }
}

} // namespace

OMStatus GRU(const float *input_data, const float *weight_input_data,
             const float *weight_hidden_data, const float *bias_input_data,
             const float *bias_hidden_data, const float *hidden_state_data, float *output_data,
             float *output_input_data, float *output_hidden_data,
             const core::OMRuntimeShape &input_shape, const core::OMRuntimeShape &output_shape,
             const core::OMRuntimeShape &weight_input_shape,
             const core::OMRuntimeShape &weight_hidden_shape)
{
  const int32_t time = input_shape.dims(0);

  core::OMRuntimeShape output_shape_fc(2);
  output_shape_fc.setDim(0, 1);
  output_shape_fc.setDim(1, weight_hidden_shape.dims(0));

  std::memcpy(output_data, hidden_state_data,
              output_shape.dims(output_shape.dimensionsCount() - 1) * sizeof(float));

  for (int i = 0; i < time; ++i)
  {
    calculateGRU(input_data, weight_input_data, weight_hidden_data, bias_input_data,
                 bias_hidden_data, output_data, input_shape, output_shape, weight_input_shape,
                 weight_hidden_shape, output_input_data, output_hidden_data, output_shape_fc);
    input_data += input_shape.dims(2);
  }
  return Ok;
}

} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_GRU_COMMON_H
