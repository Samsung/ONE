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
#include <cker/train/operation/ReLU6.h>

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
                                           IPortableTensor *back_prop_input,
                                           const IPortableTensor *back_prop_output, float alpha,
                                           float beta, ElementwiseActivationType op_type)
{
  assert(input != nullptr);
  assert(output != nullptr);
  assert(back_prop_input != nullptr);
  assert(back_prop_output != nullptr);

  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;

  _op_type = op_type;

  switch (op_type)
  {
    case ElementwiseActivationType::kReLU:
      if (input->data_type() == OperandType::FLOAT32)
      {
        if ((alpha == std::numeric_limits<float>::infinity() || alpha == 6.0f) && beta == 0.f)
        {
          cpu::ops::ElementwiseActivationLayer::configure(
            input, output, alpha, beta, cpu::ops::ElementwiseActivationType::kReLU);

          auto relu_cker = [&alpha]() {
            if (alpha == std::numeric_limits<float>::infinity())
              return nnfw::cker::train::ReLUGrad;
            else if (alpha == 6.0f)
              return nnfw::cker::train::ReLU6Grad;
            else
              throw std::runtime_error{"no supported relu kernel"};
          }();

          _backward_kernel = [relu_cker](const IPortableTensor *output,
                                         const IPortableTensor *incoming,
                                         IPortableTensor *outgoing) {
            relu_cker(getShape(output), getBuffer<float>(output), getShape(incoming),
                      getBuffer<float>(incoming), getShape(outgoing), getBuffer<float>(outgoing));
          };
        }
        else
        {
          throw std::runtime_error(
            "train ElementwiseActivationLayer : Unsupported ReLU activation type");
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
  _backward_kernel(_output, _back_prop_output, _back_prop_input);
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
