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
namespace ops
{

template <typename T> void MinLayer::minimum()
{
  nnfw::cker::Min<T>(getTensorShape(_lhs), reinterpret_cast<const T *>(_lhs->buffer()),
                     getTensorShape(_rhs), reinterpret_cast<const T *>(_rhs->buffer()),
                     getTensorShape(_output), reinterpret_cast<T *>(_output->buffer()));
}

void MinLayer::minQuant8()
{
  if (_lhs->data_scale() == _rhs->data_scale() && _lhs->data_scale() == _output->data_scale())
  {
    if (_lhs->data_offset() == _rhs->data_offset() && _lhs->data_offset() == _output->data_offset())
    {
      return nnfw::cker::Min<uint8_t>(
          getTensorShape(_lhs), reinterpret_cast<const uint8_t *>(_lhs->buffer()),
          getTensorShape(_rhs), reinterpret_cast<const uint8_t *>(_rhs->buffer()),
          getTensorShape(_output), reinterpret_cast<uint8_t *>(_output->buffer()));
    }
  }
  throw std::runtime_error("Min NYI for quantized");
}

void MinLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                         IPortableTensor *output)
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
    minimum<float>();
  }
  else if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    minQuant8();
  }
  else if (_lhs->data_type() == OperandType::INT32)
  {
    minimum<int32_t>();
  }
  else
  {
    throw std::runtime_error{"Min: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
