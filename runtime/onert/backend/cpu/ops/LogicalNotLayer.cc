/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LogicalNotLayer.h"

#include "OperationUtils.h"

#include <cker/operation/LogicalNot.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

LogicalNotLayer::LogicalNotLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void LogicalNotLayer::logicalNotBool8()
{
  nnfw::cker::LogicalNot(getTensorShape(_input), reinterpret_cast<const bool *>(_input->buffer()),
                         getTensorShape(_output), reinterpret_cast<bool *>(_output->buffer()));
}

void LogicalNotLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void LogicalNotLayer::run()
{
  if (_input->data_type() == OperandType::BOOL8)
  {
    logicalNotBool8();
  }
  else
  {
    throw std::runtime_error{"LogicalNot: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
