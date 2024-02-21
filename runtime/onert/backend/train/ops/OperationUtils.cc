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

#include "OperationUtils.h"

#include <cker/eigen/redux_functor.h>
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

const IPortableTensor *backpropActivation(const ir::Activation &activation,
                                          const IPortableTensor *output,
                                          const IPortableTensor *input_backprop,
                                          IPortableTensor *output_backprop)
{
  assert(output != nullptr);
  assert(input_backprop != nullptr);

  // handle NONE - just propagate incoming gradient
  if (activation == ir::Activation::NONE)
  {
    return input_backprop;
  }

  assert(output_backprop != nullptr);

  // handle other activation
  switch (activation)
  {
    case ir::Activation::RELU:
      nnfw::cker::train::ReLUGrad(getShape(output), getBuffer<float>(output),
                                  getShape(input_backprop), getBuffer<float>(input_backprop),
                                  getShape(output_backprop), getBuffer<float>(output_backprop));
      break;
    case ir::Activation::RELU6:
      nnfw::cker::train::ReLU6Grad(getShape(output), getBuffer<float>(output),
                                   getShape(input_backprop), getBuffer<float>(input_backprop),
                                   getShape(output_backprop), getBuffer<float>(output_backprop));
      break;
    // TODO: Add other activation backpropagation here
    default:
      throw std::runtime_error("Unsupported activation type yet");
  }
  return output_backprop;
}

void biasGrad(const IPortableTensor *input_backprop, IPortableTensor *bias_grad)
{
  assert(bias_grad);

  nnfw::cker::Shape input_backprop_shape = getShape(input_backprop);
  float *input_backprop_buffer = reinterpret_cast<float *>(input_backprop->buffer());

  nnfw::cker::Shape bias_grad_shape = getShape(bias_grad);
  float *bias_grad_buffer = getBuffer<float>(bias_grad);

  nnfw::cker::functor::biasReductionHelper(input_backprop_buffer, input_backprop_shape,
                                           bias_grad_buffer, bias_grad_shape);
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
