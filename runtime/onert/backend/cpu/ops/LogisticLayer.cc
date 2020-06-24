/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "LogisticLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Logistic.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

LogisticLayer::LogisticLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void LogisticLayer::logisticFloat32()
{
  nnfw::cker::Logistic(getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
                       getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void LogisticLayer::logisticQuant8()
{
  // cker quant8 logistic is not implemented yet
  throw std::runtime_error{"NYI"};
}

void LogisticLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void LogisticLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    logisticFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    logisticQuant8();
  }
  else
  {
    throw std::runtime_error{"Logistic: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
