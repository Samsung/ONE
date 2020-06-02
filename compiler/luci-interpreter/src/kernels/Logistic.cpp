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

#include "kernels/Logistic.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>
#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter
{
namespace kernels
{

Logistic::Logistic(const Tensor *input, Tensor *output) : _input(input), _output(output) {}

void Logistic::configure()
{
  assert(_input->element_type() == _output->element_type());
  if (_input->element_type() == DataType::U8)
    assert(_output->scale() == 1. / 256);
  _output->resize(_input->shape());
}

void Logistic::execute() const
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

void Logistic::evalFloat() const
{
  tflite::reference_ops::Logistic(getTensorShape(_input), getTensorData<float>(_input),
                                  getTensorShape(_output), getTensorData<float>(_output));
}

void Logistic::evalQuantized() const
{
  uint8_t data_table[256] = {0};
  const auto input_scale = static_cast<double>(_input->scale());
  const auto input_zero_point = static_cast<int32_t>(_input->zero_point());
  const auto output_scale = static_cast<double>(_output->scale());
  const auto output_zero_point = static_cast<int32_t>(_output->zero_point());
  const float inverse_scale = 1 / output_scale;
  int32_t maxval = std::numeric_limits<uint8_t>::max();
  int32_t minval = std::numeric_limits<uint8_t>::min();
  for (int32_t val = minval; val <= maxval; ++val)
  {
    const float dequantized = input_scale * (val - input_zero_point);
    const float transformed = 1.0f / (1.0f + std::exp(-dequantized));
    const float rescaled = std::round(transformed * inverse_scale);
    const int32_t quantized = static_cast<int32_t>(rescaled + output_zero_point);
    data_table[static_cast<uint8_t>(val)] =
        static_cast<uint8_t>(std::max(std::min(maxval, quantized), minval));
  }
  const int size = tflite::MatchingFlatSize(getTensorShape(_input), getTensorShape(_output));
  uint8_t *output_data = getTensorData<uint8_t>(_output);
  const uint8_t *input_data = getTensorData<uint8_t>(_input);
  for (int i = 0; i < size; ++i)
  {
    output_data[i] = data_table[input_data[i]];
  }
}

} // namespace kernels
} // namespace luci_interpreter
