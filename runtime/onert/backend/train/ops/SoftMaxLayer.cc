/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SoftMaxLayer.h"

#include "OperationUtils.h"

#include <cker/train/operation/SoftMax.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

SoftMaxLayer::SoftMaxLayer()
  : cpu::ops::SoftMaxLayer(), _deriv_input{nullptr}, _deriv_output{nullptr}
{
  // DO NOTHING
}

void SoftMaxLayer::configure(const IPortableTensor *input, const float beta,
                             IPortableTensor *output, IPortableTensor *deriv_input,
                             const IPortableTensor *deriv_output)
{
  cpu::ops::SoftMaxLayer::configure(input, beta, output);

  _deriv_input = deriv_input;
  _deriv_output = deriv_output;
}

void SoftMaxLayer::forward(bool) { cpu::ops::SoftMaxLayer::run(); }

void SoftMaxLayer::backward()
{
  assert(_deriv_output->data_type() == _input->data_type());
  switch (_deriv_output->data_type())
  {
    case OperandType::FLOAT32:
    {
      nnfw::cker::train::SoftMaxGrad(getShape(_output), getBuffer<float>(_output),
                                     getShape(_deriv_output), getBuffer<float>(_deriv_output),
                                     getShape(_deriv_input), getBuffer<float>(_deriv_input));
      break;
    }
    default:
      throw std::runtime_error("train SoftMaxLayer: unsupported data type");
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
