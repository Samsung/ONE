/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/MirrorPad.h"

#include "kernels/Utils.h"

#include <algorithm>

namespace luci_interpreter
{
namespace kernels
{

MirrorPad::MirrorPad(const Tensor *input, const Tensor *paddings, Tensor *output,
                     const MirrorPadParams &params)
  : KernelWithParams<MirrorPadParams>({input, paddings}, {output}, params)
{
}

void MirrorPad::configure()
{
  const Shape &input_shape = input()->shape();
  const int num_dims = input_shape.num_dims();

  if (num_dims > 4)
    throw std::runtime_error("Unsupported number of dimensions.");

  LUCI_INTERPRETER_CHECK(output()->element_type() == input()->element_type());
  LUCI_INTERPRETER_CHECK(paddings()->element_type() == DataType::S32);
  // Paddings shape should be [N, 2].
  LUCI_INTERPRETER_CHECK(paddings()->shape().num_dims() == 2);
  LUCI_INTERPRETER_CHECK(paddings()->shape().dim(0) == num_dims);
  LUCI_INTERPRETER_CHECK(paddings()->shape().dim(1) == 2);

  Shape output_shape(num_dims);
  const auto *paddings_data = getTensorData<int32_t>(paddings());
  for (int i = 0; i < num_dims; ++i)
  {
    const int32_t padding_before = paddings_data[i * 2];
    const int32_t padding_after = paddings_data[i * 2 + 1];
    LUCI_INTERPRETER_CHECK(padding_before >= 0 && padding_after >= 0);
    output_shape.dim(i) = input_shape.dim(i) + padding_before + padding_after;
  }

  output()->resize(output_shape);
}

namespace
{

// Helper method that fills the left and right pads.
template <typename T>
inline void getPadding(const T *data, int offset, int64_t *left_pad, int64_t *right_pad)
{
  *left_pad = static_cast<int64_t>(*(data + offset * 2));
  *right_pad = static_cast<int64_t>(*(data + offset * 2 + 1));
}

// Given dimension index and the left/right padding.
// Returns the corresponding dimension in the input array.
inline int getInputDimension(int padded_dimension, int left_pad, int right_pad, int input_dim_size,
                             int offset)
{
  (void)right_pad;

  if (padded_dimension < left_pad)
  {
    const int original_ind = left_pad + offset - 1;
    return original_ind - (std::min(padded_dimension, original_ind - offset));
  }
  padded_dimension -= left_pad;
  if (padded_dimension >= input_dim_size)
  {
    padded_dimension -= input_dim_size;
    const int original_ind = input_dim_size - (1 + offset);
    return original_ind - std::min(padded_dimension, original_ind);
  }
  return padded_dimension;
}

// Given and index in output array, returns the index of the value
// in input array.
int getFlatIndex(int index, int num_dims, const DataType padding_matrix_type,
                 const uint8_t *padding_matrix_data, const int32_t *input_dims,
                 int *output_dims_num_elements, int *input_dims_num_elements, const int offset)
{
  int flat_index = 0;
  int64_t left_pad = 0, right_pad = 0, dimension_index, index_in_input;

  for (int i = 0; i < num_dims; ++i)
  {
    switch (padding_matrix_type)
    {
      case DataType::S32:
        getPadding(reinterpret_cast<const int32_t *>(padding_matrix_data), i, &left_pad,
                   &right_pad);
        break;
      case DataType::S64:
        getPadding(reinterpret_cast<const int64_t *>(padding_matrix_data), i, &left_pad,
                   &right_pad);
        break;
      default:
        break;
    }
    dimension_index = index / output_dims_num_elements[i];

    index_in_input = getInputDimension(dimension_index, left_pad, right_pad, input_dims[i], offset);

    flat_index += index_in_input * input_dims_num_elements[i];
    index %= output_dims_num_elements[i];
  }

  return flat_index;
}

template <typename T>
void eval(const DataType padding_matrix_type, const uint8_t *padding_matrix_data,
          const int32_t *input_dims, int *output_dims_num_elements, int *input_dims_num_elements,
          const T *input_data, T *output_data, const int offset, const int num_dims,
          const int output_size)
{
  for (int i = 0; i < output_size; ++i)
  {
    int index = getFlatIndex(i, num_dims, padding_matrix_type, padding_matrix_data, input_dims,
                             output_dims_num_elements, input_dims_num_elements, offset);
    output_data[i] = input_data[index];
  }
}

} // namespace

void MirrorPad::execute() const
{
  const Tensor &t_input = *input();
  const Tensor &t_paddings = *paddings();
  Tensor &t_output = *output();

  const auto offset = params().mode != MirrorPadMode::REFLECT ? 0 : 1;
  const auto input_dims = t_input.shape().num_dims();
  const auto output_size = t_output.shape().num_elements();

  int output_dims_num_elements[5];
  int input_dims_num_elements[5];
  int32_t input_shape_dim[5];

  for (int i = 0; i < input_dims; i++)
  {
    output_dims_num_elements[i] = 1;
    input_dims_num_elements[i] = 1;
    input_shape_dim[i] = t_input.shape().dim(i);
  }

  for (int i = input_dims - 2; i >= 0; i--)
  {
    output_dims_num_elements[i] = output_dims_num_elements[i + 1] * t_output.shape().dim(i + 1);
    input_dims_num_elements[i] = input_dims_num_elements[i + 1] * t_input.shape().dim(i + 1);
  }

  switch (t_input.element_type())
  {
    case DataType::FLOAT32:
      eval(t_paddings.element_type(), t_paddings.data<uint8_t>(), input_shape_dim,
           output_dims_num_elements, input_dims_num_elements, t_input.data<float>(),
           t_output.data<float>(), offset, input_dims, output_size);
      break;

    case DataType::U8:
      eval(t_paddings.element_type(), t_paddings.data<uint8_t>(), input_shape_dim,
           output_dims_num_elements, input_dims_num_elements, t_input.data<uint8_t>(),
           t_output.data<uint8_t>(), offset, input_dims, output_size);
      break;

    default:
      throw std::runtime_error("luci-intp MirrorPad Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
