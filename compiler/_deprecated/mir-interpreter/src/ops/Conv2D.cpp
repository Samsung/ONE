/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Conv2D.h"
#include "QuantizationHelpers.h"
#include "Common.h"

#include "mir/Tensor.h"

#include <cmath>

namespace mir_interpreter
{

using namespace mir;

static std::int32_t calcOffset(const Shape &shape, std::int32_t i0, std::int32_t i1,
                               std::int32_t i2, std::int32_t i3)
{
  return ((i0 * shape.dim(1) + i1) * shape.dim(2) + i2) * shape.dim(3) + i3;
}

template <typename T> struct Conv2DImpl
{
  static void run(const TensorVariant &input, const TensorVariant &kernel,
                  const Conv2DOpAttributes &attributes, TensorVariant &result,
                  const TensorVariant *fused_bias);
};

template <typename T>
void Conv2DImpl<T>::run(const TensorVariant &input, const TensorVariant &kernel,
                        const Conv2DOpAttributes &attributes, TensorVariant &result,
                        const TensorVariant *fused_bias)
{
  const auto *input_data = reinterpret_cast<const T *>(input.atOffset(0));
  const auto *kernel_data = reinterpret_cast<const T *>(kernel.atOffset(0));
  auto *result_data = reinterpret_cast<T *>(result.atOffset(0));

  const Shape &input_shape = input.getShape();
  const Shape &output_shape = result.getShape();
  const Shape &kernel_shape = kernel.getShape();

  const std::vector<std::int32_t> &strides = attributes.strides;
  const std::vector<std::int32_t> &padding_before = attributes.padding_before;
  const std::int32_t num_groups = attributes.num_groups;
  assert(attributes.data_format == DataFormat::NHWC);

  const std::int32_t batch_size = output_shape.dim(0);
  const std::int32_t output_height = output_shape.dim(1);
  const std::int32_t output_width = output_shape.dim(2);
  const std::int32_t kernel_height = kernel_shape.dim(1);
  const std::int32_t kernel_width = kernel_shape.dim(2);
  const std::int32_t input_height = input_shape.dim(1);
  const std::int32_t input_width = input_shape.dim(2);

  const std::int32_t num_in_channels = input_shape.dim(3);
  const std::int32_t num_out_channels = output_shape.dim(3);

  assert(num_in_channels % num_groups == 0);
  assert(num_out_channels % num_groups == 0);

  const std::int32_t out_group_size = num_out_channels / num_groups;
  const std::int32_t in_group_size = num_in_channels / num_groups;

  assert(kernel_shape.dim(3) == in_group_size);
  assert(kernel_shape.dim(0) == num_out_channels);

  for (std::int32_t batch = 0; batch < batch_size; ++batch)
  {
    for (std::int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (std::int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (std::int32_t group = 0; group < num_groups; ++group)
        {
          const std::int32_t out_group_offset = group * out_group_size;
          const std::int32_t in_group_offset = group * in_group_size;

          for (std::int32_t out_c = 0; out_c < out_group_size; ++out_c)
          {
            const std::int32_t in_y_origin = (out_y * strides[0]) - padding_before[0];
            const std::int32_t in_x_origin = (out_x * strides[1]) - padding_before[1];

            T sum = 0.0f;

            for (std::int32_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y)
            {
              for (std::int32_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x)
              {
                for (std::int32_t in_c = 0; in_c < in_group_size; ++in_c)
                {
                  const std::int32_t in_y = in_y_origin + kernel_y;
                  const std::int32_t in_x = in_x_origin + kernel_x;

                  if ((in_y >= 0 && in_y < input_height) && (in_x >= 0 && in_x < input_width))
                  {
                    const std::int32_t in_offset =
                      calcOffset(input_shape, batch, in_y, in_x, in_group_offset + in_c);
                    const std::int32_t kernel_offset =
                      calcOffset(kernel_shape, out_group_offset + out_c, kernel_y, kernel_x, in_c);
                    const T input_val = input_data[in_offset];
                    const T kernel_val = kernel_data[kernel_offset];
                    sum += kernel_val * input_val;
                  }
                }
              }
            }

            const std::int32_t out_offset =
              calcOffset(output_shape, batch, out_y, out_x, out_group_offset + out_c);
            result_data[out_offset] = sum;
          }
        }
      }
    }
  }
}

template <> struct Conv2DImpl<uint8_t>
{
  static void run(const TensorVariant &input, const TensorVariant &kernel,
                  const Conv2DOpAttributes &attributes, TensorVariant &result,
                  const TensorVariant *fused_bias);
};

void Conv2DImpl<uint8_t>::run(const TensorVariant &input, const TensorVariant &kernel,
                              const Conv2DOpAttributes &attributes, TensorVariant &result,
                              const TensorVariant *fused_bias)
{
  if (!fused_bias)
  {
    throw std::runtime_error{"Quantized Conv2D cannot be executed without fused bias"};
  }

  const auto &input_type = input.getType();
  const auto &kernel_type = kernel.getType();
  const auto &bias_type = fused_bias->getType();
  const auto &output_type = result.getType();
  (void)bias_type;

  assert(input_type.isQuantized());
  assert(kernel_type.isQuantized());
  assert(bias_type.isQuantized());
  assert(output_type.isQuantized());
  assert(input_type.getElementType() == DataType::UINT8);
  assert(kernel_type.getElementType() == DataType::UINT8);
  assert(bias_type.getElementType() == DataType::INT32);
  assert(output_type.getElementType() == DataType::UINT8);

  int32_t input_offset = -input_type.getQuantization().getZeroPoint();
  int32_t kernel_offset = -kernel_type.getQuantization().getZeroPoint();
  int32_t output_offset = output_type.getQuantization().getZeroPoint();

  double input_scale = input_type.getQuantization().getScale();
  double kernel_scale = kernel_type.getQuantization().getScale();
  double output_scale = output_type.getQuantization().getScale();

  double real_multiplier = input_scale * kernel_scale / output_scale;
  int32_t output_multiplier = 0;
  int output_shift = 0;
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  const Shape &in_shape = input.getShape();
  const Shape &kernel_shape = kernel.getShape();
  const Shape &out_shape = result.getShape();
  const auto &strides = attributes.strides;
  const std::vector<int32_t> &pads = attributes.padding_before;
  assert(attributes.num_groups == 1);
  assert(attributes.data_format == DataFormat::NHWC);

  assert(in_shape.rank() == 4);
  assert(kernel_shape.rank() == 4);
  assert(kernel_shape.dim(3) == in_shape.dim(3));
  assert(kernel_shape.dim(0) == out_shape.dim(3));
  assert(strides.size() == 2);
  assert(pads.size() == 2);

  int32_t stride_height = strides[0];
  int32_t stride_width = strides[1];

  int32_t pad_height = pads[0];
  int32_t pad_width = pads[1];

  int32_t input_height = in_shape.dim(1);
  int32_t input_width = in_shape.dim(2);

  Tensor<uint8_t> input_accessor(input);
  Tensor<uint8_t> kernel_accessor(kernel);
  Tensor<int32_t> bias_accessor(*fused_bias);
  Tensor<uint8_t> res_accessor(result);

  int32_t output_min = std::numeric_limits<uint8_t>::min();
  int32_t output_max = std::numeric_limits<uint8_t>::max();

  for (int batch = 0; batch < out_shape.dim(0); ++batch)
  {
    for (int out_y = 0; out_y < out_shape.dim(1); ++out_y)
    {
      for (int out_x = 0; out_x < out_shape.dim(2); ++out_x)
      {
        for (int out_channel = 0; out_channel < out_shape.dim(3); ++out_channel)
        {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < kernel_shape.dim(1); ++filter_y)
          {
            for (int filter_x = 0; filter_x < kernel_shape.dim(2); ++filter_x)
            {
              for (int in_channel = 0; in_channel < kernel_shape.dim(3); ++in_channel)
              {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height))
                {
                  Index in_index{batch, in_y, in_x, in_channel};
                  Index ker_index{out_channel, filter_y, filter_x, in_channel};
                  int32_t input_val = input_accessor.at(in_index);
                  int32_t kernel_val = kernel_accessor.at(ker_index);
                  acc += (kernel_val + kernel_offset) * (input_val + input_offset);
                }
              }
            }
          }
          acc += bias_accessor.at(Index{out_channel});
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_min);
          acc = std::min(acc, output_max);
          Index out_index{batch, out_y, out_x, out_channel};
          res_accessor.at(out_index) = static_cast<uint8_t>(acc);
        }
      }
    }
  }
}

void Conv2D(const mir::TensorVariant &input, const mir::TensorVariant &kernel,
            const mir::Conv2DOpAttributes &attributes, mir::TensorVariant &result,
            const mir::TensorVariant *fused_bias)
{
  dispatch<Conv2DImpl>(result.getElementType(), input, kernel, attributes, result, fused_bias);
}

} // namespace mir_interpreter
