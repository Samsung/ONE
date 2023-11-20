/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/PadV2.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/pad.h>

#include <limits>

namespace luci_interpreter
{
namespace kernels
{

PadV2::PadV2(const Tensor *input, const Tensor *paddings, const Tensor *constant_values,
             Tensor *output)
  : Kernel({input, paddings, constant_values}, {output})
{
}

void PadV2::configure()
{
  const Shape &input_shape = input()->shape();
  const int num_dims = input_shape.num_dims();

  if (num_dims > 4)
    throw std::runtime_error("Unsupported number of dimensions.");

  assert(output()->element_type() == input()->element_type());
  assert(paddings()->element_type() == DataType::S32);
  assert(constant_values()->element_type() == output()->element_type());
  // Paddings shape should be [N, 2].
  assert(paddings()->shape().num_dims() == 2);
  assert(paddings()->shape().dim(0) == num_dims);
  assert(paddings()->shape().dim(1) == 2);
  // Constant values elements number should be 1.
  assert(constant_values()->shape().num_elements() == 1);

  Shape output_shape(num_dims);
  const auto *paddings_data = getTensorData<int32_t>(paddings());
  for (int i = 0; i < num_dims; ++i)
  {
    const int32_t padding_before = paddings_data[i * 2];
    const int32_t padding_after = paddings_data[i * 2 + 1];
    assert(padding_before >= 0 && padding_after >= 0);
    output_shape.dim(i) = input_shape.dim(i) + padding_before + padding_after;
  }

  output()->resize(output_shape);
}

void PadV2::execute() const
{
  const int num_dims = input()->shape().num_dims();

  tflite::PadParams params{};
  params.left_padding_count = num_dims;
  params.right_padding_count = num_dims;

  const auto *paddings_data = getTensorData<int32_t>(paddings());
  for (int i = num_dims - 1; i >= 0; --i)
  {
    params.left_padding[i] = paddings_data[i * 2];
    params.right_padding[i] = paddings_data[i * 2 + 1];
  }

  switch (input()->element_type())
  {
    case DataType::FLOAT32:
    {
      const auto pad_value = getTensorData<float>(constant_values())[0];
      tflite::reference_ops::Pad(params, getTensorShape(input()), getTensorData<float>(input()),
                                 &pad_value, getTensorShape(output()),
                                 getTensorData<float>(output()));
      break;
    }
    case DataType::U8:
    {
      assert(output()->zero_point() >= std::numeric_limits<uint8_t>::min());
      assert(output()->zero_point() <= std::numeric_limits<uint8_t>::max());
      const auto pad_value = getTensorData<uint8_t>(constant_values())[0];
      tflite::reference_ops::Pad(params, getTensorShape(input()), getTensorData<uint8_t>(input()),
                                 &pad_value, getTensorShape(output()),
                                 getTensorData<uint8_t>(output()));
      break;
    }
    default:
      throw std::runtime_error("luci-intp PadV2 Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
