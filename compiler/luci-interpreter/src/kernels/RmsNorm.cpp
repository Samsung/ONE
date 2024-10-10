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

#include "kernels/RmsNorm.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/common.h>
#include <cmath>

namespace luci_interpreter
{
namespace kernels
{

RmsNorm::RmsNorm(const Tensor *input, const Tensor *gamma, Tensor *output,
                 const RmsNormParams &params)
  : KernelWithParams<RmsNormParams>({input, gamma}, {output}, params)
{
}

void RmsNorm::configure()
{
  auto num_dims = input()->shape().num_dims();
  LUCI_INTERPRETER_CHECK(num_dims == 3 || num_dims == 4);
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(gamma()->element_type() == input()->element_type());
  LUCI_INTERPRETER_CHECK(gamma()->shape().num_dims() == 1);
  LUCI_INTERPRETER_CHECK((gamma()->shape().dim(0) == input()->shape().dim(num_dims - 1)) ||
                         (gamma()->shape().dim(0) == 1));

  output()->resize(input()->shape());
}

void RmsNorm::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("luci-intp RmsNorm Unsupported type.");
  }
}

void RmsNorm::evalFloat() const
{
  tflite::RuntimeShape input_shape = getTensorShape(input());
  auto output_shape = getTensorShape(output());

  const float *input_data = getTensorData<float>(input());
  const float *gamma_data = getTensorData<float>(gamma());
  auto gamma_shape = getTensorShape(gamma());
  bool single_gamma = gamma_shape.DimensionsCount() == 1 && gamma_shape.Dims(0) == 1;
  float *output_data = getTensorData<float>(output());

  if (input_shape.DimensionsCount() == 4)
  {
    // Dimensions for image case are (N x H x W x C)
    const int32_t batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
    const int32_t heights = tflite::MatchingDim(input_shape, 1, output_shape, 1);
    const int32_t widths = tflite::MatchingDim(input_shape, 2, output_shape, 2);
    const int32_t channels = tflite::MatchingDim(input_shape, 3, output_shape, 3);
    for (int32_t batch = 0; batch < batches; batch++)
    {
      for (int32_t height = 0; height < heights; height++)
      {
        for (int32_t width = 0; width < widths; width++)
        {
          double square_sum = 0.0f;
          for (int32_t channel = 0; channel < channels; channel++)
          {
            double input_val =
              input_data[tflite::Offset(input_shape, batch, height, width, channel)];
            square_sum += (input_val * input_val);
          }
          double rms = std::sqrt((square_sum / channels) + params().epsilon);
          for (int32_t channel = 0; channel < channels; channel++)
          {
            double gamma = single_gamma ? gamma_data[0] : gamma_data[channel];
            output_data[tflite::Offset(output_shape, batch, height, width, channel)] =
              gamma *
              (input_data[tflite::Offset(input_shape, batch, height, width, channel)] / rms);
          }
        }
      }
    }
  }
  else if (input_shape.DimensionsCount() == 3)
  {
    // Dimensions for non image case are (N x C x D1 x D2 … Dn)
    const int32_t batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
    const int32_t channels = tflite::MatchingDim(input_shape, 1, output_shape, 1);
    const int32_t size = tflite::MatchingDim(input_shape, 2, output_shape, 2);
    for (int32_t batch = 0; batch < batches; batch++)
    {
      for (int32_t channel = 0; channel < channels; channel++)
      {
        double square_sum = 0.0f;
        size_t offset =
          static_cast<size_t>(batch * channels * size) + static_cast<size_t>(channel * size);
        for (int32_t i = 0; i < size; i++)
        {
          double input_val = input_data[offset + i];
          square_sum += (input_val * input_val);
        }
        double rms = std::sqrt((square_sum / size) + params().epsilon);
        for (int32_t i = 0; i < size; i++)
        {
          double gamma = single_gamma ? gamma_data[0] : gamma_data[i];
          output_data[offset + i] = gamma * (input_data[offset + i] / rms);
        }
      }
    }
  }
  else
    throw std::runtime_error("luci-intp RmsNorm unsupported rank.");
}

} // namespace kernels
} // namespace luci_interpreter
