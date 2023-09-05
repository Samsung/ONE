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

#include "ElementwiseActivationLayer.h"

#include "OperationUtils.h"

#include <cker/train/operation/ReLU.h>

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
                                           IPortableTensor *deriv_input,
                                           const IPortableTensor *deriv_output, float alpha,
                                           float beta, ElementwiseActivationType op_type)
{
  assert(input != nullptr);
  assert(output != nullptr);
  assert(deriv_input != nullptr);
  assert(deriv_output != nullptr);

  _deriv_input = deriv_input;
  _deriv_output = deriv_output;

  _op_type = op_type;

  switch (op_type)
  {
    case ElementwiseActivationType::kReLU:
      if (input->data_type() == OperandType::FLOAT32)
      {
        if (alpha == std::numeric_limits<float>::infinity() && beta == 0.f)
        {
          cpu::ops::ElementwiseActivationLayer::configure(
            input, output, alpha, beta, cpu::ops::ElementwiseActivationType::kReLU);

          _backward_kernel = [](const IPortableTensor *output, const IPortableTensor *incoming,
                                IPortableTensor *outgoing) {
            nnfw::cker::train::ReLUGrad(getShape(output), getBuffer<float>(output),
                                        getShape(incoming), getBuffer<float>(incoming),
                                        getShape(outgoing), getBuffer<float>(outgoing));
          };
        }
        else
        {
          throw std::runtime_error("train ElementwiseActivationLayer : This layer does not "
                                   "suppport other ReLU except for ReLU(0-inf)");
        }
      }
      else
      {
        throw std::runtime_error("train ElementwiseActivationLayer: Unsupported datatype");
      }
      break;
    default:
      throw std::runtime_error("train ElementwiseActivationLayer: Unsupported activation type yet");
  }
}

void ElementwiseActivationLayer::forward(bool) { cpu::ops::ElementwiseActivationLayer::run(); }

void ElementwiseActivationLayer::backward()
{
  _backward_kernel(_output, _deriv_output, _deriv_input);
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
