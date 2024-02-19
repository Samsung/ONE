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

#include <cker/operation/Reduce.h>
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
  // TODO Use optimized kernel
  assert(bias_grad);
  std::vector<int32_t> axes{0, 1, 2};
  nnfw::cker::Reduce reduce_kernel;

  reduce_kernel.prepare(input_backprop->getShape().rank(), axes.size());
  bool result = reduce_kernel.ReduceGeneric<float>(
    getShape(input_backprop), getBuffer<float>(input_backprop), getShape(bias_grad),
    getBuffer<float>(bias_grad), axes, false /* keep_dims */, 0.f,
    [](const float current, const float in) -> float { return in + current; });
  if (!result)
    throw std::runtime_error{"train DepthwiseConvolutionLayer: Fail to calculate bias gradient"};
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
