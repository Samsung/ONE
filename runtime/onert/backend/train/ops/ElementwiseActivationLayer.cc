/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ElementwiseActivationLayer.h"

#include <ops/OperationUtils.h>
#include <cker/train/ReLU.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

ElementwiseActivationLayer::ElementwiseActivationLayer() : cpu::ops::ElementwiseActivationLayer()
{
  // DO NOTHING
}

void ElementwiseActivationLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                                           const IPortableTensor *grad_input,
                                           IPortableTensor *grad_output, float alpha, float beta,
                                           ElementwiseActivationType op_type)
{
  assert(grad_input != nullptr);
  assert(grad_output != nullptr);

  _grad_input = grad_input;
  _grad_output = grad_output;

  _op_type = op_type;

  switch (op_type)
  {
    case ElementwiseActivationType::kReLU:
      cpu::ops::ElementwiseActivationLayer::configure(input, output, alpha, beta,
                                                      cpu::ops::ElementwiseActivationType::kReLU);
      break;
  }
}

void ElementwiseActivationLayer::forward(bool) { cpu::ops::ElementwiseActivationLayer::run(); }

void ElementwiseActivationLayer::backward()
{
  // TODO Implement this
  switch (_op_type)
  {
    case ElementwiseActivationType::kReLU:
      if (_input->data_type() == OperandType::FLOAT32)
      {
        nnfw::cker::train::ReLUDeriv(
          cpu::ops::getShape(_grad_input), cpu::ops::getBuffer<float>(_grad_input),
          cpu::ops::getShape(_grad_output), cpu::ops::getBuffer<float>(_grad_output));
      }
      else
      {
        throw std::runtime_error{"ElementwiseActivationLayer(ReLU): unsupported data type"};
      }
      break;
    default:
      throw std::runtime_error("ElementwiseActivationLayer: unsupported op type");
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
