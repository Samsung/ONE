/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RmsNormLayer.h"

#include "OperationUtils.h"

#include <cker/operation/RmsNorm.h>
#include <cker/Types.h>

namespace onert::backend::cpu::ops
{

void RmsNormLayer::configure(const IPortableTensor *input, const IPortableTensor *gamma,
                             float epsilon, IPortableTensor *output)
{
  assert(input != nullptr);
  assert(output != nullptr);

  _input = input;
  _output = output;
  _gamma = gamma;
  _epsilon = epsilon;
}

void RmsNormLayer::run()
{
  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::RmsNormParams param;
      param.epsilon = _epsilon;
      nnfw::cker::RmsNorm(param, getShape(_input), getBuffer<float>(_input), getShape(_gamma),
                          getBuffer<float>(_gamma), getShape(_output), getBuffer<float>(_output));
      break;

    default:
      throw std::runtime_error{"RmsNorm: Unsupported data type"};
  }
}

} // namespace onert::backend::cpu::ops
