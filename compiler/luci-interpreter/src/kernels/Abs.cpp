/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Abs.h"

#include "kernels/Utils.h"

#include <cmath> // abs for float

namespace luci_interpreter
{
namespace kernels
{

Abs::Abs(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Abs::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());

  output()->resize(input()->shape());
}

void Abs::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      eval<float>();
      break;
    default:
      throw std::runtime_error("luci-intp Abs Unsupported type.");
  }
}

template <typename T> void Abs::eval() const
{
  const auto *input_data = input()->data<T>();
  auto *output_data = output()->data<T>();

  const int size = tflite::MatchingFlatSize(getTensorShape(input()), getTensorShape(output()));

  for (int i = 0; i < size; ++i)
  {
    output_data[i] = std::abs(input_data[i]);
  }
}

} // namespace kernels
} // namespace luci_interpreter
