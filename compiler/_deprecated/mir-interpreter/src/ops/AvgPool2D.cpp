/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "AvgPool2D.h"
#include "Common.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

namespace mir_interpreter
{

using namespace mir;

template <typename T> class AvgPool2DImpl
{
public:
  static void run(const mir::ops::AvgPool2DOp &op, const mir::TensorVariant &input_var,
                  mir::TensorVariant &output);
};

template <typename T>
void AvgPool2DImpl<T>::run(const ops::AvgPool2DOp &op, const TensorVariant &input_var,
                           TensorVariant &output)
{
  const auto &input_shape = op.getInputShape(0);
  const auto &output_shape = op.getOutputShape(0);
  const auto &window_size = op.getWindowSize();
  const auto &strides = op.getStrides();
  const auto &padding_before = op.getPaddingBefore();
  const auto &padding_after = op.getPaddingAfter();
  (void)padding_after;

  constexpr int num_spatial_dims = 2;
  assert(input_var.getShape().rank() == 4);
  assert(window_size.size() == num_spatial_dims);
  assert(strides.size() == num_spatial_dims);
  assert(padding_before.size() == num_spatial_dims);
  assert(padding_after.size() == num_spatial_dims);

  Tensor<T> res_accessor(output);
  Tensor<T> input(input_var);

  ShapeRange in_range(input_shape);
  Index in_index(input_shape.rank());

  for (const auto &out_index : ShapeRange(output_shape))
  {
    T result = 0;
    size_t num_elements = 0;

    // Assuming NHWC format.
    in_index.at(0) = out_index.at(0);
    in_index.at(3) = out_index.at(3);

    for (const auto &window_index : ShapeRange(Shape(window_size)))
    {
      // Assuming NHWC format.
      for (int i = 0; i < num_spatial_dims; ++i)
        in_index.at(1 + i) =
          out_index.at(1 + i) * strides[i] + window_index.at(i) - padding_before[i];

      if (in_range.contains(in_index))
      {
        num_elements++;
        result += input.at(in_index);
      }
      else if (op.getIncludePad())
      {
        num_elements++;
      }
    }

    result /= num_elements;
    res_accessor.at(out_index) = result;
  }
}

template <> struct AvgPool2DImpl<uint8_t>
{
  static void run(const mir::ops::AvgPool2DOp &op, const mir::TensorVariant &input,
                  mir::TensorVariant &output);
};

void AvgPool2DImpl<uint8_t>::run(const ops::AvgPool2DOp &op, const TensorVariant &input,
                                 TensorVariant &output)
{
  const auto &input_type = input.getType();
  const auto &output_type = op.getOutput(0)->getType();
  (void)input_type;

  assert(input_type.isQuantized());
  assert(output_type.isQuantized());
  assert(input_type.getElementType() == DataType::UINT8);

  const auto &input_shape = op.getInputShape(0);
  const auto &output_shape = op.getOutputShape(0);
  const auto &window_size = op.getWindowSize();
  const auto &strides = op.getStrides();
  const auto &padding_before = op.getPaddingBefore();
  const auto &padding_after = op.getPaddingAfter();
  (void)padding_after;

  constexpr int num_spatial_dims = 2;
  assert(input.getShape().rank() == 4);
  assert(window_size.size() == num_spatial_dims);
  assert(strides.size() == num_spatial_dims);
  assert(padding_before.size() == num_spatial_dims);
  assert(padding_after.size() == num_spatial_dims);

  Tensor<uint8_t> input_accessor(input);
  Tensor<uint8_t> res_accessor(output);

  ShapeRange in_range(input_shape);
  Index in_index(input_shape.rank());

  int32_t output_min = std::numeric_limits<uint8_t>::min();
  int32_t output_max = std::numeric_limits<uint8_t>::max();

  for (const auto &out_index : ShapeRange(output_shape))
  {
    int32_t result = 0;
    size_t num_elements = 0;

    // Assuming NHWC format.
    in_index.at(0) = out_index.at(0);
    in_index.at(3) = out_index.at(3);

    for (const auto &window_index : ShapeRange(Shape(window_size)))
    {
      // Assuming NHWC format.
      for (int i = 0; i < num_spatial_dims; ++i)
        in_index.at(1 + i) =
          out_index.at(1 + i) * strides[i] + window_index.at(i) - padding_before[i];

      if (in_range.contains(in_index))
      {
        num_elements++;
        result += input_accessor.at(in_index);
      }
      else if (op.getIncludePad())
      {
        num_elements++;
      }
    }
    result = (result + num_elements / 2) / num_elements;
    result = std::max(result, output_min);
    result = std::min(result, output_max);
    res_accessor.at(out_index) = static_cast<uint8_t>(result);
  }
}

void AvgPool2D(const mir::ops::AvgPool2DOp &op, const mir::TensorVariant &input,
               mir::TensorVariant &output)
{
  dispatch<AvgPool2DImpl>(output.getElementType(), op, input, output);
}

} // namespace mir_interpreter
