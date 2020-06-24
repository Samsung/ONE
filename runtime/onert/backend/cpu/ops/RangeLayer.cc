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

#include "RangeLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Range.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
RangeLayer::RangeLayer() : _start(nullptr), _limit(nullptr), _delta(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void RangeLayer::configure(const IPortableTensor *start, const IPortableTensor *limit,
                           const IPortableTensor *delta, IPortableTensor *output)
{
  _start = start;
  _limit = limit;
  _delta = delta;
  _output = output;
}

void RangeLayer::run()
{
  switch (_output->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::Range<float>(reinterpret_cast<float *>(_start->buffer()),
                               reinterpret_cast<float *>(_limit->buffer()),
                               reinterpret_cast<float *>(_delta->buffer()),
                               reinterpret_cast<float *>(_output->buffer()));
      break;
    case OperandType::INT32:
      nnfw::cker::Range<int32_t>(reinterpret_cast<int32_t *>(_start->buffer()),
                                 reinterpret_cast<int32_t *>(_limit->buffer()),
                                 reinterpret_cast<int32_t *>(_delta->buffer()),
                                 reinterpret_cast<int32_t *>(_output->buffer()));
      break;
    default:
      throw std::runtime_error{"Range: unsupported data type"};
      break;
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
