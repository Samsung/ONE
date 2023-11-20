/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Log.h"
#include "kernels/Utils.h"

#include <cmath>

namespace luci_interpreter
{

namespace kernels
{

Log::Log(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Log::configure() { output()->resize(input()->shape()); }

void Log::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("luci-intp Log Unsupported type.");
  }
}

void Log::evalFloat() const
{
  const auto input_data = getTensorData<float>(input());
  const auto input_shape = input()->shape();
  auto output_data = getTensorData<float>(output());
  for (int64_t i = 0; i < input_shape.num_elements(); ++i)
  {
    output_data[i] = std::log(input_data[i]);
  }
}

} // namespace kernels
} // namespace luci_interpreter
