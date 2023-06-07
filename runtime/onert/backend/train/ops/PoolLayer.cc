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

#include "PoolLayer.h"

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

PoolLayer::PoolLayer() : cpu::ops::PoolLayer()
{
  // DO NOTHING
}

void PoolLayer::configure(const IPortableTensor *input, const uint32_t paddingLeft,
                          const uint32_t paddingRight, const uint32_t paddingTop,
                          const uint32_t paddingBottom, const uint32_t strideWidth,
                          const uint32_t strideHeight, const uint32_t kernelWidth,
                          const uint32_t kernelHeight, const ir::Activation activation,
                          IPortableTensor *output, const PoolType op_type)
{
  switch (op_type)
  {
    case PoolType::kMax:
      cpu::ops::PoolLayer::configure(input, paddingLeft, paddingRight, paddingTop, paddingBottom,
                                     strideWidth, strideHeight, kernelWidth, kernelHeight,
                                     activation, output, cpu::ops::PoolType::kMax);
      break;
    default:
      throw std::runtime_error("PoolLayer: Unsupported pool type");
  }
}

void PoolLayer::forward(bool training)
{
  if (training)
  {
    // TODO Implement details
  }
  else
  {
    cpu::ops::PoolLayer::run();
  }
}

void PoolLayer::backward()
{
  // TODO Implement this
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
