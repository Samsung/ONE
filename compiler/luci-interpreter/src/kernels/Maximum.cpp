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

#include "kernels/Maximum.h"

#include "kernels/Utils.h"

namespace luci_interpreter
{
namespace kernels
{

Maximum::Maximum(const Tensor *input1, const Tensor *input2, Tensor *output)
    : Kernel({input1, input2}, {output})
{
}

void Maximum::configure()
{
  LUCI_INTERPRETER_CHECK(input1()->element_type() == input2()->element_type())
  LUCI_INTERPRETER_CHECK(input1()->element_type() == output()->element_type())
  output()->resize(input1()->shape());
}

void Maximum::execute() const
{
  switch (input1()->element_type())
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

void Maximum::evalFloat() const
{
  const int size = tflite::MatchingFlatSize(getTensorShape(input1()), getTensorShape(output()));
  float *output_data = getTensorData<float>(output());
  const float *input_data1 = getTensorData<float>(input1());
  const float *input_data2 = getTensorData<float>(input2());
  for (int i = 0; i < size; ++i)
  {
    output_data[i] = std::max(input_data1[i], input_data2[i]);
  }
}

void Maximum::evalQuantized() const
{
  const int size = tflite::MatchingFlatSize(getTensorShape(input1()), getTensorShape(output()));
  uint8_t *output_data = getTensorData<uint8_t>(output());
  const uint8_t *input_data1 = getTensorData<uint8_t>(input1());
  const uint8_t *input_data2 = getTensorData<uint8_t>(input2());
  for (int i = 0; i < size; ++i)
  {
    output_data[i] = std::max(input_data1[i], input_data2[i]);
  }
}

} // namespace kernels
} // namespace luci_interpreter
