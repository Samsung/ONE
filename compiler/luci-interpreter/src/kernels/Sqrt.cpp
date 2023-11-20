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

#include "kernels/Sqrt.h"
#include "kernels/Utils.h"

#include <stdexcept>
#include <cmath>

namespace luci_interpreter
{

namespace kernels
{

Sqrt::Sqrt(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Sqrt::configure()
{
  if (input()->element_type() != output()->element_type())
  {
    throw std::runtime_error("Input/output tensor data type mismatch.");
  }
  output()->resize(input()->shape());
}

void Sqrt::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;

    default:
      throw std::runtime_error("luci-intp Sqrt Unsupported type.");
  }
}

void Sqrt::evalFloat() const
{
  auto in = getTensorData<float>(input());
  auto out = getTensorData<float>(output());
  auto size = getTensorShape(input()).FlatSize();
  for (auto i = in; i != in + size; ++i)
  {
    *out = std::sqrt(*i);
    ++out;
  }
}

} // namespace kernels
} // namespace luci_interpreter
