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
  : cpu::ops::SoftMaxLayer(), _back_prop_input{nullptr}, _back_prop_output{nullptr}
{
  // DO NOTHING
}

void SoftMaxLayer::configure(const IPortableTensor *input, const float beta,
                             IPortableTensor *output, IPortableTensor *back_prop_input,
                             const IPortableTensor *back_prop_output)
{
  cpu::ops::SoftMaxLayer::configure(input, beta, output);

  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;
}

void SoftMaxLayer::forward(bool) { cpu::ops::SoftMaxLayer::run(); }

void SoftMaxLayer::backward()
{
  assert(_back_prop_output->data_type() == _input->data_type());
  switch (_back_prop_output->data_type())
  {
    case OperandType::FLOAT32:
    {
      nnfw::cker::train::SoftMaxGrad(
        getShape(_output), getBuffer<float>(_output), getShape(_back_prop_output),
        getBuffer<float>(_back_prop_output), getShape(_back_prop_input),
        getBuffer<float>(_back_prop_input));
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
