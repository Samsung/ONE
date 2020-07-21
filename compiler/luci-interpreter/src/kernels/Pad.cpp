/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Pad.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

namespace luci_interpreter
{
namespace kernels
{

Pad::Pad(const Tensor *input, const Tensor *paddings, Tensor *output)
    : _input(input), _paddings(paddings), _output(output)
{
}

void Pad::configure()
{
  const Shape &input_shape = _input->shape();
  const int num_dims = input_shape.num_dims();

  if (num_dims > 4)
    throw std::runtime_error("Unsupported number of dimensions.");

  assert(_output->element_type() == _input->element_type());
  assert(_paddings->element_type() == DataType::S32);
  // Paddings shape should be [N, 2].
  assert(_paddings->shape().num_dims() == 2);
  assert(_paddings->shape().dim(0) == num_dims);
  assert(_paddings->shape().dim(1) == 2);

  Shape output_shape(num_dims);
  const auto *paddings_data = getTensorData<int32_t>(_paddings);
  for (int i = 0; i < num_dims; ++i)
  {
    const int32_t padding_before = paddings_data[i * 2];
    const int32_t padding_after = paddings_data[i * 2 + 1];
    assert(padding_before >= 0 && padding_after >= 0);
    output_shape.dim(i) = input_shape.dim(i) + padding_before + padding_after;
  }

  _output->resize(output_shape);
}

void Pad::execute() const
{
  const int num_dims = _input->shape().num_dims();

  tflite::PadParams params{};
  params.left_padding_count = num_dims;
  params.right_padding_count = num_dims;

  const auto *paddings_data = getTensorData<int32_t>(_paddings);
  for (int i = num_dims - 1; i >= 0; --i)
  {
    params.left_padding[i] = paddings_data[i * 2];
    params.right_padding[i] = paddings_data[i * 2 + 1];
  }

  switch (_input->element_type())
  {
    case DataType::FLOAT32:
    {
      const float pad_value = 0.0f;
      tflite::reference_ops::Pad(params, getTensorShape(_input), getTensorData<float>(_input),
                                 &pad_value, getTensorShape(_output),
                                 getTensorData<float>(_output));
      break;
    }
    case DataType::U8:
    {
      assert(_output->zero_point() >= std::numeric_limits<uint8_t>::min());
      assert(_output->zero_point() <= std::numeric_limits<uint8_t>::max());
      const auto pad_value = static_cast<uint8_t>(_output->zero_point());
      tflite::reference_ops::Pad(params, getTensorShape(_input), getTensorData<uint8_t>(_input),
                                 &pad_value, getTensorShape(_output),
                                 getTensorData<uint8_t>(_output));
      break;
    }
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
