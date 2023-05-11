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

#include "ReverseLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Reverse.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

void ReverseLayer::run()
{

  if (_axis->total_size() != 4)
  {
    throw std::runtime_error{"Reverse: only support 1 axis"};
  }
  int32_t axis = *getBuffer<int32_t>(_axis);
  if (axis < 0)
  {
    axis += _input->getShape().rank();
  }

  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::Reverse<float>(axis, getShape(_input), getBuffer<float>(_input),
                                 getShape(_output), getBuffer<float>(_output));
      break;
    default:
      throw std::runtime_error{"Reverse: unsupported data type"};
  }
}

void ReverseLayer::configure(const IPortableTensor *input, const IPortableTensor *axis,
                             IPortableTensor *output)
{
  _input = input;
  _axis = axis;
  _output = output;
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
