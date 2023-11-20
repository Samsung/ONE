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

#include "kernels/LogicalNot.h"

#include "kernels/Utils.h"

#include "kernels/BinaryOpCommon.h"

namespace luci_interpreter
{
namespace kernels
{

LogicalNot::LogicalNot(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void LogicalNot::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  output()->resize(input()->shape());
}

void LogicalNot::execute() const
{
  switch (input()->element_type())
  {
    case DataType::BOOL:
      evalLogicalNot();
      break;
    default:
      throw std::runtime_error("luci-intp LogicalNot Unsupported type.");
  }
}

inline void LogicalNot::evalLogicalNot() const
{
  const int size = tflite::MatchingFlatSize(getTensorShape(input()), getTensorShape(output()));
  bool *output_data = getTensorData<bool>(output());
  const bool *input_data = getTensorData<bool>(input());
  for (int i = 0; i < size; ++i)
  {
    output_data[i] = !input_data[i];
  }
}

} // namespace kernels
} // namespace luci_interpreter
