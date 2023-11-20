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

#include "kernels/Relu.h"
#include "kernels/Utils.h"

#include "PALRelu.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Relu::Relu(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Relu::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  if (input()->element_type() == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && output()->zero_point() == 0);
  }

  if (input()->element_type() == DataType::U8 || input()->element_type() == DataType::S16)
  {
    double multiplier = input()->scale() / output()->scale();
    quantizeMultiplier(multiplier, &_output_multiplier, &_output_shift);
  }
  output()->resize(input()->shape());
}

void Relu::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    case DataType::S16:
      evalQuantizedS16();
      break;
    default:
      throw std::runtime_error("luci-intp Relu Unsupported type.");
  }
}

void Relu::evalFloat() const
{
  const auto input_data = getTensorData<float>(input());
  const auto input_shape = getTensorShape(input());
  auto output_data = getTensorData<float>(output());
  auto output_shape = getTensorShape(output());

  luci_interpreter_pal::Relu(input_shape, input_data, output_shape, output_data);
}

void Relu::evalQuantized() const
{
  tflite::ReluParams params;
  params.input_offset = input()->zero_point();
  params.output_offset = output()->zero_point();
  params.output_multiplier = _output_multiplier;
  params.output_shift = _output_shift;

  params.quantized_activation_min =
    std::max(static_cast<int32_t>(std::numeric_limits<uint8_t>::min()), params.output_offset);
  params.quantized_activation_max = static_cast<int32_t>(std::numeric_limits<uint8_t>::max());

  luci_interpreter_pal::ReluX(params, getTensorShape(input()), getTensorData<uint8_t>(input()),
                              getTensorShape(output()), getTensorData<uint8_t>(output()));
}

void Relu::evalQuantizedS16() const
{
  const auto *input_data = getTensorData<int16_t>(input());
  auto *output_data = getTensorData<int16_t>(output());

  constexpr int32_t output_min = 0;
  constexpr int32_t output_max = std::numeric_limits<int16_t>::max();

  const int32_t num_elements = input()->shape().num_elements();

  for (int32_t i = 0; i < num_elements; ++i)
  {
    const int32_t input_val = input_data[i];
    int32_t output_val =
      tflite::MultiplyByQuantizedMultiplier(input_val, _output_multiplier, _output_shift);
    output_val = std::max(output_val, output_min);
    output_val = std::min(output_val, output_max);
    output_data[i] = static_cast<int16_t>(output_val);
  }
}

} // namespace kernels
} // namespace luci_interpreter
