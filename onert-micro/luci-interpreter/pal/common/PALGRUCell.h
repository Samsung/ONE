/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_GRU_H
#define LUCI_INTERPRETER_PAL_GRU_H

#include "PALUtils.h"
#include "ProcessBroadcastShapes.h"
#include "PALFullyConnected.h"
#include "PALLogistic.h"

namespace luci_interpreter_pal
{

namespace
{
void calculateGRU(const float *input_data, const float *weight_input_data,
                  const float *weight_hidden_data, const float *bias_input_data,
                  const float *bias_hidden_data, float *output_data,
                  const int32_t *input_shape, const int32_t *output_shape, const int32_t *weight_input_shape,
                  const int32_t *weight_hidden_shape,
                  float *output_input_data,
                  float *output_hidden_data, const int32_t *output_shape_fc)
{
  // Calculate FC for hidden (output)
  FullyConnectedParams op_params{};
  float activation_min{};
  float activation_max{};
  luci_interpreter::kernels::calculateActivationRange(luci_interpreter::FusedActFunc::NONE,
                                    &activation_min, &activation_max);

  luci_interpreter_pal::FullyConnectedParams params{};
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;

  FullyConnected(op_params, output_shape, output_data, weight_hidden_shape, weight_hidden_data,
                 bias_hidden_data, output_shape_fc, output_hidden_data);

  // Calcuate FC for input
  FullyConnected(op_params, input_shape, input_data, weight_input_shape, weight_input_data,
                 bias_input_data, output_shape_fc, output_input_data);

  int num_elements = output_shape_fc[1] / 3;

  float *second_hidden_part = output_hidden_data + num_elements;
  float *second_input_part = output_input_data + num_elements;

  float *third_hidden_part = second_hidden_part + num_elements;
  float *third_input_part = second_input_part + num_elements;

  // Calculate Left part
  for (int i = 0; i < num_elements; ++i)
  {
    output_hidden_data[i] += output_input_data[i];
  }
  Logistic(num_elements, output_hidden_data, output_hidden_data);

  // Calculate most left add
  float *most_left_part_final = output_hidden_data;
  float *first_part = output_hidden_data;
  for (int i = 0; i < num_elements; ++i)
  {
    output_data[i] *= most_left_part_final[i];
    first_part[i] = 1.0f - first_part[i];
  }


  // Clalc third part
  float *third_part = third_hidden_part;
  for (int i = 0; i < num_elements; ++i)
  {
    third_part[i] += third_input_part[i];
  }
  Logistic(num_elements, third_part, third_part);

  for (int i = 0; i < num_elements; ++i)
  {
    third_part[i] *= second_hidden_part[i];
    third_part[i] += second_input_part[i];
    third_part[i] = std::tanh(third_part[i]);
    third_part[i] *= first_part[i];
    output_data[i] += third_part[i];
  }
}

} // namespace

void GRU(float *input_data, const float *weight_input_data,
         const float *weight_hidden_data, const float *bias_input_data,
         const float *bias_hidden_data, const float *hidden_state_data, float *output_data,
         const int32_t *input_shape, const int32_t *output_shape, const int32_t *weight_input_shape,
         const int32_t *weight_hidden_shape)
{
  const int32_t time = input_shape[0];
  input_shape += 1;

  auto output_input_data = std::make_unique<float []>(weight_hidden_shape[0]);
  auto output_hidden_data = std::make_unique<float []>(weight_hidden_shape[0]);

  int32_t output_shape_fc[] = {1, 96};

  std::memcpy(output_data, hidden_state_data, output_shape[1]);

  for (int i = 0; i < time; ++i)
  {
    // input_shape should be (1, 6)
    calculateGRU(input_data, weight_input_data, weight_hidden_data,
                 bias_input_data, bias_hidden_data, output_data, input_shape,
                 output_shape, weight_input_shape, weight_hidden_shape, output_input_data.get(),
                 output_hidden_data.get(), output_shape_fc);
    auto tmp = input_shape[1];
    input_data += input_shape[1];
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_GRU_H
