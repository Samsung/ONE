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

#ifndef LUCI_INTERPRETER_PAL_GRU_H
#define LUCI_INTERPRETER_PAL_GRU_H

#include <tensorflow/lite/kernels/internal/reference/fully_connected.h>
#include "PALreference_ops.h"
namespace luci_interpreter_pal
{

// tflite's Logistic does not provide inplace Logistic kernel
void Logistic(const int flat_size, const float *input_data, float *output_data)
{
  const float cutoff_upper = 16.619047164916992188f;
  const float cutoff_lower = -9.f;

  // Rational for using approximation in reference kernel.
  // 0. This approximation gives enough precision for float.
  // 1. This works around an issue on an embedded chipset where exp() does not
  // return correctly as expected - exp(x) should return inf when overflown
  // not 1.701417   IEEE 754 defines representation for inf.
  // 2. This will speed up calculation and is matching the behavior in the
  // optimized kernels. (check the definition of scalar_logistic_op<float>)

  for (int i = 0; i < flat_size; i++)
  {
    float val = input_data[i];
    float result;
    if (val > cutoff_upper)
    {
      result = 1.0f;
    }
    else if (val < cutoff_lower)
    {
      result = std::exp(val);
    }
    else
    {
      result = 1.f / (1.f + std::exp(-val));
    }
    output_data[i] = result;
  }
}

void calculateGRU(const float *input_data, const float *weight_input_data,
                  const float *weight_hidden_data, const float *bias_input_data,
                  const float *bias_hidden_data, float *output_data,
                  const tflite::RuntimeShape &input_shape, const tflite::RuntimeShape &output_shape,
                  const tflite::RuntimeShape &weight_input_shape,
                  const tflite::RuntimeShape &weight_hidden_shape, float *output_input_data,
                  float *output_hidden_data, const tflite::RuntimeShape &output_shape_fc)
{
  tflite::FullyConnectedParams op_params{};
  // As FC nodes doesn't have any activations inside GRU, let' use just numeric limits
  op_params.float_activation_min = std::numeric_limits<float>::lowest();
  op_params.float_activation_max = std::numeric_limits<float>::max();

  // FC Input
  tflite::RuntimeShape bias_input_shape{weight_input_shape.Dims(0)};
  tflite::reference_ops::FullyConnected(op_params, output_shape, output_data, weight_input_shape,
                                        weight_input_data, bias_input_shape, bias_input_data,
                                        output_shape_fc, output_input_data);

  // FC Hidden
  tflite::RuntimeShape bias_hidden_shape{weight_hidden_shape.Dims(0)};
  // Note: input for this FC node will be saved without intermediate buffer
  tflite::reference_ops::FullyConnected(op_params, input_shape, input_data, weight_hidden_shape,
                                        weight_hidden_data, bias_hidden_shape, bias_hidden_data,
                                        output_shape_fc, output_hidden_data);

  int num_elements = output_shape_fc.Dims(1) / 3;

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

  // Calculate most left mul
  float *most_left_part_final = output_input_data;
  float *first_part = output_input_data;
  for (int i = 0; i < num_elements; ++i)
  {
    output_data[i] *= most_left_part_final[i];
    first_part[i] = 1.0f - first_part[i];
  }

  // Calc second part
  for (int i = 0; i < num_elements; ++i)
  {
    second_hidden_part[i] += second_input_part[i];
  }

  Logistic(num_elements, second_hidden_part, second_hidden_part);

  for (int i = 0; i < num_elements; ++i)
  {
    second_hidden_part[i] *= third_input_part[i];
    second_hidden_part[i] += third_hidden_part[i];
  }

  for (int i = 0; i < num_elements; ++i)
  {
    if (second_hidden_part[i] > 19)
    {
      second_hidden_part[i] = 1;
    }
    else if (second_hidden_part[i] < -19)
    {
      second_hidden_part[i] = -1;
    }
    else
    {
      second_hidden_part[i] = std::tanh(second_hidden_part[i]);
    }
  }

  for (int i = 0; i < num_elements; ++i)
  {
    second_hidden_part[i] *= first_part[i];
    output_data[i] += second_hidden_part[i];
  }
}

void GRU(const float *input_data, const float *weight_input_data, const float *weight_hidden_data,
         const float *bias_input_data, const float *bias_hidden_data,
         const float *hidden_state_data, float *output_data, float *output_input_data,
         float *output_hidden_data, const tflite::RuntimeShape &input_shape,
         const tflite::RuntimeShape &output_shape, const tflite::RuntimeShape &weight_input_shape,
         const tflite::RuntimeShape &weight_hidden_shape)
{
  const int32_t time = input_shape.Dims(0);

  tflite::RuntimeShape output_shape_fc(2);
  output_shape_fc.SetDim(0, 1);
  output_shape_fc.SetDim(1, weight_hidden_shape.Dims(0));

  std::memcpy(output_data, hidden_state_data, output_shape.FlatSize() * sizeof(float));

  for (int i = 0; i < time; ++i)
  {
    calculateGRU(input_data, weight_input_data, weight_hidden_data, bias_input_data,
                 bias_hidden_data, output_data, input_shape, output_shape, weight_input_shape,
                 weight_hidden_shape, output_input_data, output_hidden_data, output_shape_fc);
    input_data += input_shape.Dims(2);
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_GRU_H
