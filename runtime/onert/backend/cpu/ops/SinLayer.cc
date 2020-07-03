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

#include "SinLayer.h"
#include "OperationUtils.h"

#include <cker/operation/Elementwise.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
SinLayer::SinLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void SinLayer::sinFloat32()
{
  nnfw::cker::Sin(getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
                  getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void SinLayer::sinQuant8() { throw std::runtime_error{"NYI"}; }

void SinLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void SinLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    sinFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    sinQuant8();
  }
  else
  {
    throw std::runtime_error{"Sin: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
