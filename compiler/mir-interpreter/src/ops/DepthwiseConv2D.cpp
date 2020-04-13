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

#include "DepthwiseConv2D.h"
#include "QuantizationHelpers.h"
#include "Common.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include <cmath>

namespace mir_interpreter
{

using namespace mir;

template <typename T> struct DepthwiseConv2DImpl
{
  static void run(const mir::ops::DepthwiseConv2DOp &op, const mir::TensorVariant &inputv,
                  const mir::TensorVariant &kernelv, const mir::TensorVariant *biasv,
                  mir::TensorVariant &output);
};

template <typename T>
void DepthwiseConv2DImpl<T>::run(const mir::ops::DepthwiseConv2DOp &op,
                                 const mir::TensorVariant &inputv,
                                 const mir::TensorVariant &kernelv, const mir::TensorVariant *biasv,
                                 mir::TensorVariant &output)
{
  const Shape &in_shape = op.getInputShape(0);
  const Shape &kernel_shape = op.getInputShape(1);
  const Shape &out_shape = op.getOutputShape(0);
  const auto &strides = op.getStrides();
  const std::vector<int32_t> &pads = op.getPaddingBefore();

  assert(in_shape.rank() == 4);
  assert(kernel_shape.rank() == 4);
  assert(kernel_shape.dim(2) == in_shape.dim(3));
  assert(in_shape.dim(3) * kernel_shape.dim(3) == out_shape.dim(3));
  assert(strides.size() == 2);
  assert(pads.size() == 2);

  int32_t channel_multiplier = kernel_shape.dim(3);

  Tensor<T> res_accessor(output);
  Tensor<T> input(inputv);
  Tensor<T> bias(*biasv);
  Tensor<T> kernel(kernelv);

  ShapeRange in_range(in_shape);
  ShapeRange kernel_range(kernel_shape);
  ShapeRange out_range(Shape{out_shape.dim(0), out_shape.dim(1), out_shape.dim(2), 1});

  Index in_index;
  in_index.resize(4);

  erase<T>(output);

  for (const auto &out_index : out_range)
  {
    Index out_index_k = out_index;
    for (const auto &kernel_index : kernel_range)
    {
      in_index.at(0) = out_index.at(0);
      for (int i = 0; i < 2; ++i)
        in_index.at(1 + i) = out_index.at(1 + i) * strides[i] + kernel_index.at(i) - pads[i];
      in_index.at(3) = kernel_index.at(2);

      if (in_range.contains(in_index))
      {
        out_index_k.at(3) = kernel_index.at(2) * channel_multiplier + kernel_index.at(3);
        res_accessor.at(out_index_k) += input.at(in_index) * kernel.at(kernel_index);
      }
    }
  }
}

template <> struct DepthwiseConv2DImpl<uint8_t>
{
  static void run(const mir::ops::DepthwiseConv2DOp &op, const mir::TensorVariant &inputv,
                  const mir::TensorVariant &kernelv, const mir::TensorVariant *biasv,
                  mir::TensorVariant &output);
};

void DepthwiseConv2DImpl<uint8_t>::run(const mir::ops::DepthwiseConv2DOp &op,
                                       const mir::TensorVariant &inputv,
                                       const mir::TensorVariant &kernelv,
                                       const mir::TensorVariant *biasv, mir::TensorVariant &output)
{
  if (!biasv)
  {
    throw std::runtime_error{"Unsupported quantized DepthwiseConv2D without fused bias"};
  }

  const auto &input_type = inputv.getType();
  const auto &kernel_type = kernelv.getType();
  const auto &bias_type = biasv->getType();
  const auto &output_type = op.getOutput(0)->getType();
  (void)bias_type;

  assert(input_type.isQuantized());
  assert(kernel_type.isQuantized());
  assert(bias_type.isQuantized());
  assert(output_type.isQuantized());
  assert(input_type.getElementType() == DataType::UINT8);
  assert(kernel_type.getElementType() == DataType::UINT8);
  assert(bias_type.getElementType() == DataType::INT32);

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

  const Shape &in_shape = inputv.getShape();
  const Shape &kernel_shape = kernelv.getShape();
  const Shape &out_shape = op.getOutputShape(0);
  const auto &strides = op.getStrides();
  const std::vector<int32_t> &pads = op.getPaddingBefore();

  assert(in_shape.rank() == 4);
  assert(kernel_shape.rank() == 4);
  assert(kernel_shape.dim(2) == in_shape.dim(3)); // HWIO
  assert(in_shape.dim(3) * kernel_shape.dim(3) == out_shape.dim(3));
  assert(strides.size() == 2);
  assert(pads.size() == 2);

  int32_t stride_height = strides[0];
  int32_t stride_width = strides[1];

  int32_t pad_height = pads[0];
  int32_t pad_width = pads[1];

  int32_t input_height = in_shape.dim(1);
  int32_t input_width = in_shape.dim(2);

  Tensor<uint8_t> input_accessor(inputv);
  Tensor<uint8_t> kernel_accessor(kernelv);
  Tensor<int32_t> bias_accessor(*biasv);
  Tensor<uint8_t> res_accessor(output);

  int32_t output_min = std::numeric_limits<uint8_t>::min();
  int32_t output_max = std::numeric_limits<uint8_t>::max();

  int batches = out_shape.dim(0);
  int output_height = out_shape.dim(1);
  int output_width = out_shape.dim(2);
  int input_depth = in_shape.dim(3);

  int filter_height = kernel_shape.dim(0); // HWIO
  int filter_width = kernel_shape.dim(1);  // HWIO

  for (int b = 0; b < batches; ++b)
  {
    for (int out_y = 0; out_y < output_height; ++out_y)
    {
      for (int out_x = 0; out_x < output_width; ++out_x)
      {
        for (int ic = 0; ic < input_depth; ++ic)
        {
          const int oc = ic;
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x)
            {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              // If the location is outside the bounds of the input image,
              // use zero as a default value.
              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height))
              {
                Index in_index{b, in_y, in_x, ic};
                Index ker_index{filter_y, filter_x, oc, 0}; // HWIO
                int32_t input_val = input_accessor.at(in_index);
                int32_t kernel_val = kernel_accessor.at(ker_index);
                acc += (kernel_val + kernel_offset) * (input_val + input_offset);
              }
            }
          }
          acc += bias_accessor.at(Index{oc});
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_min);
          acc = std::min(acc, output_max);
          Index out_index{b, out_y, out_x, oc};
          res_accessor.at(out_index) = static_cast<uint8_t>(acc);
        }
      }
    }
  }
}

void DepthwiseConv2D(const mir::ops::DepthwiseConv2DOp &op, const mir::TensorVariant &input,
                     const mir::TensorVariant &kernel, mir::TensorVariant &output,
                     const mir::TensorVariant *bias)
{
  dispatch<DepthwiseConv2DImpl>(output.getElementType(), op, input, kernel, bias, output);
}

} // namespace mir_interpreter
