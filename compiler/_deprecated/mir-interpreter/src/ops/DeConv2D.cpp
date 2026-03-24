/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DeConv2D.h"
#include "Common.h"

#include "mir/TensorUtil.h"

#include <cstdint>

namespace mir_interpreter
{

using namespace mir;
using std::int32_t;

static int32_t calcOffset(const Shape &shape, int32_t i0, int32_t i1, int32_t i2, int32_t i3)
{
  return ((i0 * shape.dim(1) + i1) * shape.dim(2) + i2) * shape.dim(3) + i3;
}

template <typename T> struct DeConv2DImpl
{
  static void run(const TensorVariant &input, const TensorVariant &kernel,
                  const Deconv2DOpAttributes &attributes, TensorVariant &output);
};

template <typename T>
void DeConv2DImpl<T>::run(const TensorVariant &input, const TensorVariant &kernel,
                          const Deconv2DOpAttributes &attributes, TensorVariant &output)
{
  // [H, W, Co, Ci] -> [Ci, H, W, Co]
  TensorVariant transposed_kernel = transposeTensor<3, 0, 1, 2>(kernel);

  const auto *input_data = reinterpret_cast<const T *>(input.atOffset(0));
  const auto *kernel_data = reinterpret_cast<const T *>(transposed_kernel.atOffset(0));
  auto *output_data = reinterpret_cast<T *>(output.atOffset(0));

  const Shape &input_shape = input.getShape();
  const Shape &output_shape = output.getShape();
  const Shape &kernel_shape = transposed_kernel.getShape();

  const std::vector<int32_t> &strides = attributes.strides;
  const std::vector<int32_t> &padding_before = attributes.padding_before;
  assert(attributes.data_format == DataFormat::NHWC);

  const int32_t batch_size = output_shape.dim(0);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);
  const int32_t kernel_height = kernel_shape.dim(1);
  const int32_t kernel_width = kernel_shape.dim(2);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);

  const int32_t num_in_channels = input_shape.dim(3);
  const int32_t num_out_channels = output_shape.dim(3);

  assert(kernel_shape.dim(0) == num_in_channels);
  assert(kernel_shape.dim(3) == num_out_channels);

  erase<T>(output);

  for (int32_t batch = 0; batch < batch_size; ++batch)
  {
    for (int32_t in_y = 0; in_y < input_height; ++in_y)
    {
      for (int32_t in_x = 0; in_x < input_width; ++in_x)
      {
        for (int32_t in_c = 0; in_c < num_in_channels; ++in_c)
        {
          const T input_val = input_data[calcOffset(input_shape, batch, in_y, in_x, in_c)];
          const int32_t out_y_origin = in_y * strides[0] - padding_before[0];
          const int32_t out_x_origin = in_x * strides[1] - padding_before[1];

          for (int32_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y)
          {
            for (int32_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x)
            {
              const int32_t out_y = out_y_origin + kernel_y;
              const int32_t out_x = out_x_origin + kernel_x;

              if ((out_y >= 0 && out_y < output_height) && (out_x >= 0 && out_x < output_width))
              {
                for (int32_t out_c = 0; out_c < num_out_channels; ++out_c)
                {
                  const int32_t kernel_offset =
                    calcOffset(kernel_shape, in_c, kernel_y, kernel_x, out_c);
                  const int32_t output_offset =
                    calcOffset(output_shape, batch, out_y, out_x, out_c);
                  const T kernel_val = kernel_data[kernel_offset];
                  output_data[output_offset] += input_val * kernel_val;
                }
              }
            }
          }
        }
      }
    }
  }
}

void DeConv2D(const TensorVariant &input, const TensorVariant &kernel,
              const Deconv2DOpAttributes &attributes, TensorVariant &output)
{
  dispatch<DeConv2DImpl>(output.getElementType(), input, kernel, attributes, output);
}

} // namespace mir_interpreter
