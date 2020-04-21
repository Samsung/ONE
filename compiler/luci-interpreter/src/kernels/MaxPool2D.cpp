/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/MaxPool2D.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

MaxPool2D::MaxPool2D(const Tensor *input, Tensor *output, const MaxPool2DParams &params)
    : _input(input), _output(output), _params(params)
{
}

void MaxPool2D::configure()
{
  const Shape &input_shape = _input->shape();
  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t depth = input_shape.dim(3);

  const int32_t output_height = computeOutputSize(_params.padding, input_height,
                                                  _params.filter_height, _params.stride_height);
  const int32_t output_width =
      computeOutputSize(_params.padding, input_width, _params.filter_width, _params.stride_width);

  _padding_height =
      computePadding(_params.stride_height, 1, input_height, _params.filter_height, output_height);
  _padding_width =
      computePadding(_params.stride_width, 1, input_width, _params.filter_width, output_width);

  _output->resize({batches, output_height, output_width, depth});
}

void MaxPool2D::execute() const
{
  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

// https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/lite/kernels/internal/reference/pooling.h
void MaxPool2D::evalFloat() const
{
  const auto *input_data = _input->data<float>();
  auto *output_data = _output->data<float>();

  const Shape &input_shape = _input->shape();
  const Shape &output_shape = _output->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t depth = input_shape.dim(3);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  for (int32_t batch = 0; batch < batches; ++batch)
  {
    for (int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (int32_t channel = 0; channel < depth; ++channel)
        {
          const int32_t in_y_origin = (out_y * _params.stride_height) - _padding_height;
          const int32_t in_x_origin = (out_x * _params.stride_width) - _padding_width;
          const int32_t filter_y_start = std::max(0, -in_y_origin);
          const int32_t filter_y_end = std::min(_params.filter_height, input_height - in_y_origin);
          const int32_t filter_x_start = std::max(0, -in_x_origin);
          const int32_t filter_x_end = std::min(_params.filter_width, input_width - in_x_origin);
          float max_val = 0.0f;
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
          {
            for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x)
            {
              const int32_t in_y = in_y_origin + filter_y;
              const int32_t in_x = in_x_origin + filter_x;
              const float val = input_data[offset(input_shape, batch, in_y, in_x, channel)];
              if (val > max_val)
              {
                max_val = val;
              }
            }
          }
          output_data[offset(output_shape, batch, out_y, out_x, channel)] =
              activationFunctionWithMinMax(max_val, activation_min, activation_max);
        }
      }
    }
  }
}

// https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/lite/kernels/internal/reference/pooling.h
void MaxPool2D::evalQuantized() const
{
  const auto *input_data = _input->data<uint8_t>();
  auto *output_data = _output->data<uint8_t>();

  const Shape &input_shape = _input->shape();
  const Shape &output_shape = _output->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t depth = input_shape.dim(3);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, _output, &activation_min, &activation_max);

  for (int32_t batch = 0; batch < batches; ++batch)
  {
    for (int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (int32_t channel = 0; channel < depth; ++channel)
        {
          const int32_t in_y_origin = (out_y * _params.stride_height) - _padding_height;
          const int32_t in_x_origin = (out_x * _params.stride_width) - _padding_width;
          const int32_t filter_y_start = std::max(0, -in_y_origin);
          const int32_t filter_y_end = std::min(_params.filter_height, input_height - in_y_origin);
          const int32_t filter_x_start = std::max(0, -in_x_origin);
          const int32_t filter_x_end = std::min(_params.filter_width, input_width - in_x_origin);
          int32_t max_val = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y)
          {
            for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x)
            {
              const int32_t in_y = in_y_origin + filter_y;
              const int32_t in_x = in_x_origin + filter_x;
              const int32_t val = input_data[offset(input_shape, batch, in_y, in_x, channel)];
              if (val > max_val)
              {
                max_val = val;
              }
            }
          }
          output_data[offset(output_shape, batch, out_y, out_x, channel)] =
              activationFunctionWithMinMax(max_val, activation_min, activation_max);
        }
      }
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
