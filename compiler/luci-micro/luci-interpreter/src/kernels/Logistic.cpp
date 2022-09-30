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

#include <tensorflow/lite/kernels/internal/reference/logistic.h>

namespace luci_interpreter
{
namespace kernels
{

Logistic::Logistic(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Logistic::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  if (input()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(output()->scale() == 1. / 256);
  }
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(input()->shape());
}

void Logistic::execute() const
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

void Logistic::evalFloat() const
{
  if (_is_inplace)
    output()->set_data_buffer(const_cast<uint8_t *>(input()->data<uint8_t>()));

  tflite::reference_ops::Logistic(getTensorShape(input()), getTensorData<float>(input()),
                                  getTensorShape(output()), getTensorData<float>(output()));
  if (_is_inplace)
  {
    auto input_tensor = const_cast<Tensor *>(input());
    input_tensor->set_data_buffer(nullptr);
  }
}

void Logistic::evalQuantized() const
{
  if (_is_inplace)
    output()->set_data_buffer(const_cast<uint8_t *>(input()->data<uint8_t>()));

  tflite::reference_ops::Logistic(getTensorShape(input()), getTensorData<int8_t>(input()),
                                  input()->scale(), input()->zero_point(), getTensorShape(output()),
                                  getTensorData<int8_t>(output()), output()->scale(),
                                  output()->zero_point());
  if (_is_inplace)
  {
    auto input_tensor = const_cast<Tensor *>(input());
    input_tensor->set_data_buffer(nullptr);
  }
}

} // namespace kernels
} // namespace luci_interpreter
