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

#include "ReLU6Layer.h"

#include "OperationUtils.h"

#include <cker/operation/ReLU6.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

ReLU6Layer::ReLU6Layer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void ReLU6Layer::relu6Float32()
{
  nnfw::cker::ReLU6(getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
                    reinterpret_cast<float *>(_output->buffer()));
}

void ReLU6Layer::relu6Quant8()
{
  // cker quant8 relu is not implemented yet
  throw std::runtime_error{"NYI"};
}

void ReLU6Layer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

void ReLU6Layer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    relu6Float32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    relu6Quant8();
  }
  else
  {
    throw std::runtime_error{"ReLU6: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
