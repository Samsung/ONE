/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MaxPool2D.h"
#include "Common.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include <limits>

namespace mir_interpreter
{

using namespace mir;

template <typename T> struct MaxPool2DImpl
{
  static void run(const mir::TensorVariant &inputv, const mir::ops::MaxPool2DOp &op,
                  mir::TensorVariant &result);
};

template <typename T>
void MaxPool2DImpl<T>::run(const TensorVariant &inputv, const ops::MaxPool2DOp &op,
                           TensorVariant &result)
{
  const auto &input_shape = op.getInputShape(0);
  const auto &output_shape = op.getOutputShape(0);
  const auto &window_size = op.getWindowSize();
  const auto &strides = op.getStrides();
  const auto &padding_before = op.getPaddingBefore();
  const auto &padding_after = op.getPaddingAfter();
  (void)padding_after;

  Tensor<T> input(inputv);

  constexpr int num_spatial_dims = 2;
  assert(input.getShape().rank() == 4);
  assert(window_size.size() == num_spatial_dims);
  assert(strides.size() == num_spatial_dims);
  assert(padding_before.size() == num_spatial_dims);
  assert(padding_after.size() == num_spatial_dims);

  Tensor<T> res_accessor(result);

  ShapeRange in_range(input_shape);
  Index in_index(input_shape.rank());

  for (const auto &out_index : ShapeRange(output_shape))
  {
    T result = std::numeric_limits<T>::lowest();

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
        result = std::max(result, input.at(in_index));
      }
    }

    res_accessor.at(out_index) = result;
  }
}

template <> struct MaxPool2DImpl<uint8_t>
{
  static void run(const mir::TensorVariant &input, const mir::ops::MaxPool2DOp &op,
                  mir::TensorVariant &result);
};

void MaxPool2DImpl<uint8_t>::run(const TensorVariant &input, const ops::MaxPool2DOp &op,
                                 TensorVariant &result)
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

  TensorType res_type(mir::DataType::UINT8, output_shape, output_type.getQuantization());
  TensorVariant res(res_type);
  Tensor<uint8_t> res_accessor(res);

  ShapeRange in_range(input_shape);
  Index in_index(input_shape.rank());

  for (const auto &out_index : ShapeRange(output_shape))
  {
    // Assuming NHWC format.
    in_index.at(0) = out_index.at(0);
    in_index.at(3) = out_index.at(3);

    uint8_t result = 0;
    for (const auto &window_index : ShapeRange(Shape(window_size)))
    {
      // Assuming NHWC format.
      for (int i = 0; i < num_spatial_dims; ++i)
        in_index.at(1 + i) =
          out_index.at(1 + i) * strides[i] + window_index.at(i) - padding_before[i];

      if (in_range.contains(in_index))
      {
        result = std::max(result, input_accessor.at(in_index));
      }
    }
    res_accessor.at(out_index) = result;
  }
}

void MaxPool2D(const mir::TensorVariant &input, const mir::ops::MaxPool2DOp &op,
               mir::TensorVariant &result)
{
  dispatch<MaxPool2DImpl>(input.getElementType(), input, op, result);
};

} // namespace mir_interpreter
