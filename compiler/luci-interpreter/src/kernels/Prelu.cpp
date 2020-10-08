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

#include "kernels/Prelu.h"

#include "kernels/BinaryOpCommon.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Prelu::Prelu(const Tensor *input, const Tensor *alpha, Tensor *output)
    : Kernel({input, alpha}, {output})
{
}

void Prelu::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(alpha()->element_type() == output()->element_type());

  if (input()->element_type() == DataType::U8 || input()->element_type() == DataType::S16)
  {
    if (input()->element_type() == DataType::S16)
    {
      LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && alpha()->zero_point() == 0 &&
                             output()->zero_point() == 0);
    }
    double alpha_multiplier = input()->scale() * alpha()->scale() / output()->scale();
    quantizeMultiplier(alpha_multiplier, &_output_multiplier_alpha, &_output_shift_alpha);
    double identity_multiplier = input()->scale() / output()->scale();
    quantizeMultiplier(identity_multiplier, &_output_multiplier_identity, &_output_shift_identity);
  }
  output()->resize(calculateShapeForBroadcast(input()->shape(), alpha()->shape()));
}

void Prelu::execute() const
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
      throw std::runtime_error("Unsupported type.");
  }
}

void Prelu::evalFloat() const
{
  const auto input_data = getTensorData<float>(input());
  const auto alpha_data = getTensorData<float>(alpha());
  const auto size = getTensorShape(input()).FlatSize();
  auto output_data = getTensorData<float>(output());

  auto PreluFunc = [](float input, float alpha) { return input >= 0.0 ? input : input * alpha; };

  if (input()->shape() != alpha()->shape())
  {
    tflite::reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
        getTensorShape(input()), getTensorData<float>(input()), getTensorShape(alpha()),
        getTensorData<float>(alpha()), getTensorShape(output()), getTensorData<float>(output()),
        PreluFunc);
  }
  else
  {
    for (auto i = decltype(size){0}; i < size; ++i)
    {
      if (input_data[i] >= 0)
        output_data[i] = input_data[i];
      else
        output_data[i] = input_data[i] * alpha_data[i];
    }
  }
}

void Prelu::evalQuantized() const
{
  tflite::PreluParams op_params{};

  op_params.input_offset = -input()->zero_point(); // Note the '-'.
  op_params.alpha_offset = -alpha()->zero_point(); // Note the '-'.
  op_params.output_offset = output()->zero_point();
  op_params.output_shift_1 = _output_shift_identity;
  op_params.output_multiplier_1 = _output_multiplier_identity;
  op_params.output_shift_2 = _output_shift_alpha;
  op_params.output_multiplier_2 = _output_multiplier_alpha;

  if (input()->shape() != alpha()->shape())
  {
    tflite::reference_ops::BroadcastPrelu4DSlow(
        op_params, getTensorShape(input()), getTensorData<uint8_t>(input()),
        getTensorShape(alpha()), getTensorData<uint8_t>(alpha()), getTensorShape(output()),
        getTensorData<uint8_t>(output()));
  }
  else
  {
    tflite::reference_ops::Prelu<uint8_t>(op_params, getTensorShape(input()),
                                          getTensorData<uint8_t>(input()), getTensorShape(alpha()),
                                          getTensorData<uint8_t>(alpha()), getTensorShape(output()),
                                          getTensorData<uint8_t>(output()));
  }
}

void Prelu::evalQuantizedS16() const
{
  constexpr int32_t quantized_min = std::numeric_limits<int16_t>::min();
  constexpr int32_t quantized_max = std::numeric_limits<int16_t>::max();

  auto fn = [this, quantized_min, quantized_max](int16_t input_val, int16_t alpha_val) {
    const int32_t output_val =
        input_val >= 0
            ? tflite::MultiplyByQuantizedMultiplier(input_val, _output_multiplier_identity,
                                                    _output_shift_identity)
            : tflite::MultiplyByQuantizedMultiplier(input_val * alpha_val, _output_multiplier_alpha,
                                                    _output_shift_alpha);
    const int32_t clamped_output = std::min(quantized_max, std::max(quantized_min, output_val));
    return static_cast<int16_t>(clamped_output);
  };

  BinaryOpBroadcastSlow(getTensorShape(input()), getTensorData<int16_t>(input()),
                        getTensorShape(alpha()), getTensorData<int16_t>(alpha()),
                        getTensorShape(output()), getTensorData<int16_t>(output()), fn);
}

} // namespace kernels
} // namespace luci_interpreter
