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

#include "LogicalOrLayer.h"

#include "OperationUtils.h"

#include <cker/operation/LogicalOr.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
void LogicalOrLayer::lorBool8()
{
  if (!HaveSameShapes(_lhs, _rhs))
  {
    nnfw::cker::LogicalOrBroadcast<bool>(
        getTensorShape(_lhs), reinterpret_cast<const bool *>(_lhs->buffer()), getTensorShape(_rhs),
        reinterpret_cast<const bool *>(_rhs->buffer()), getTensorShape(_output),
        reinterpret_cast<bool *>(_output->buffer()));
  }
  else
  {
    nnfw::cker::LogicalOrElementwise<bool>(getTensorShape(_lhs),
                                           reinterpret_cast<const bool *>(_lhs->buffer()),
                                           reinterpret_cast<const bool *>(_rhs->buffer()),
                                           reinterpret_cast<bool *>(_output->buffer()));
  }
}

void LogicalOrLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                               IPortableTensor *output)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);

  _lhs = lhs;
  _rhs = rhs;
  _output = output;
}

void LogicalOrLayer::run()
{
  if ((_lhs->data_type() == OperandType::BOOL8) && (_rhs->data_type() == OperandType::BOOL8))
  {
    lorBool8();
  }
  else
  {
    throw std::runtime_error{"LogicalOr: Unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
