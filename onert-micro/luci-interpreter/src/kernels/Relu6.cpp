/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Relu6.h"
#include "kernels/Utils.h"

#include "PALRelu6.h"

namespace luci_interpreter
{

namespace kernels
{

Relu6::Relu6(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Relu6::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());

  if (input()->element_type() == DataType::U8)
  {
    double multiplier = input()->scale() / output()->scale();
    quantizeMultiplier(multiplier, &_output_multiplier, &_output_shift);
  }
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(input()->shape());
}

void Relu6::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

void Relu6::evalFloat() const
{
  const auto input_data = getTensorData<float>(input());
  const auto input_shape = getTensorShape(input());
  auto output_data = getTensorData<float>(output());
  auto output_shape = getTensorShape(output());

  luci_interpreter_pal::Relu6(input_shape, input_data, output_shape, output_data);
}

void Relu6::evalQuantized() const
{
  tflite::ReluParams params;
  params.input_offset = input()->zero_point();
  params.output_offset = output()->zero_point();
  params.output_multiplier = _output_multiplier;
  params.output_shift = _output_shift;

  params.quantized_activation_min =
    std::max(static_cast<int32_t>(std::numeric_limits<uint8_t>::min()), params.output_offset);
  params.quantized_activation_max =
    std::min(static_cast<int32_t>(std::numeric_limits<uint8_t>::max()),
             params.output_offset + static_cast<int32>(roundf(6.f / output()->scale())));

  luci_interpreter_pal::ReluX(params, getTensorShape(input()), getTensorData<uint8_t>(input()),
                              getTensorShape(output()), getTensorData<uint8_t>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
