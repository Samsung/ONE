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

#include "LogLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Elementwise.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

LogLayer::LogLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void LogLayer::logFloat32()
{
  nnfw::cker::Log(convertTensorToCkerShape(_input),
                  reinterpret_cast<const float *>(_input->buffer()),
                  convertTensorToCkerShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void LogLayer::logQuant8() { throw std::runtime_error{"NYI"}; }

void LogLayer::configure(const ITensor *input, ITensor *output)
{
  _input = input;
  _output = output;
}

void LogLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    logFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT8_ASYMM)
  {
    logQuant8();
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
