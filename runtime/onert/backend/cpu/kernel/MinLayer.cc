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

#include "MinLayer.h"

#include "OperationUtils.h"

#include <cker/operation/MaxMin.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

void MinLayer::minFloat32()
{
  nnfw::cker::Min<float>(
      convertTensorToCkerShape(_lhs), reinterpret_cast<const float *>(_lhs->buffer()),
      convertTensorToCkerShape(_rhs), reinterpret_cast<const float *>(_rhs->buffer()),
      convertTensorToCkerShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void MinLayer::minQuant8()
{
  // TODO Check whether cker for quant8 min produces correct results
  // nnfw::cker::Min<uint8_t>(
  //     convertTensorToCkerShape(_lhs), reinterpret_cast<const uint8_t*>(_lhs->buffer()),
  //     convertTensorToCkerShape(_rhs), reinterpret_cast<const uint8_t*>(_rhs->buffer()),
  //     convertTensorToCkerShape(_output), reinterpret_cast<uint8_t*>(_output->buffer()));

  throw std::runtime_error("Min NYI for quantized");
}

void MinLayer::configure(const ITensor *lhs, const ITensor *rhs, ITensor *output)
{
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);

  _lhs = lhs;
  _rhs = rhs;
  _output = output;
}

void MinLayer::run()
{
  if (_lhs->data_type() == OperandType::FLOAT32)
  {
    minFloat32();
  }
  else if (_lhs->data_type() == OperandType::QUANT8_ASYMM)
  {
    minQuant8();
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
