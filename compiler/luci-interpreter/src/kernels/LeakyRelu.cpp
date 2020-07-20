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

#include "kernels/LeakyRelu.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>
#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

LeakyRelu::LeakyRelu(const Tensor *input, Tensor *output, const LeakyReluParams &params)
    : KernelWithParams<LeakyReluParams>({input}, {output}, params)
{
}

void LeakyRelu::configure()
{
  assert(input()->element_type() == output()->element_type());
  if (input()->element_type() == DataType::U8)
  {
    _q_alpha = static_cast<uint8_t>(std::max<float>(
        std::numeric_limits<uint8_t>::min(),
        std::min<float>(std::numeric_limits<uint8_t>::max(),
                        std::round(input()->zero_point() + (params().alpha / input()->scale())))));
    double real_multiplier = input()->scale() * input()->scale() / output()->scale();
    quantizeMultiplierSmallerThanOneExp(real_multiplier, &_output_multiplier, &_output_shift);
  }
  output()->resize(input()->shape());
}

void LeakyRelu::execute() const
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
      throw std::runtime_error("Unsupported type.");
  }
}

void LeakyRelu::evalFloat() const
{
  tflite::LeakyReluParams op_params{};
  op_params.alpha = params().alpha;
  tflite::optimized_ops::LeakyRelu(op_params, getTensorShape(input()),
                                   getTensorData<float>(input()), getTensorShape(output()),
                                   getTensorData<float>(output()));
}

void LeakyRelu::evalQuantized() const
{
  tflite::LeakyReluParams op_params{};
  op_params.input_offset = input()->zero_point();
  op_params.alpha_offset = input()->zero_point();
  op_params.output_offset = output()->zero_point();

  op_params.output_multiplier = _output_multiplier;
  op_params.output_shift = _output_shift;

  tflite::reference_ops::QuantizeLeakyRelu(
      op_params, _q_alpha, getTensorShape(input()), getTensorData<uint8_t>(input()),
      getTensorShape(output()), getTensorData<uint8_t>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
