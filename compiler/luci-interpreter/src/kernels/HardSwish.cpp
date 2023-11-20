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

#include "kernels/HardSwish.h"
#include "kernels/Utils.h"

#include "PALHardSwish.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

HardSwish::HardSwish(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void HardSwish::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  output()->resize(input()->shape());
}

void HardSwish::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      luci_interpreter_pal::HardSwish(getTensorShape(input()), getTensorData<float>(input()),
                                      getTensorShape(output()), getTensorData<float>(output()));
      break;
    default:
      throw std::runtime_error("luci-intp HardSwish Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
