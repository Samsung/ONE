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

#include "PoolLayer.h"

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

PoolLayer::PoolLayer() : cpu::ops::PoolLayer()
{
  // DO NOTHING
}

#define POOLING_PARAMETERS                              \
  nnfw::cker::PoolParams op_params;                     \
  op_params.stride_height = strideHeight;               \
  op_params.stride_width = strideWidth;                 \
  op_params.filter_height = kernelHeight;               \
  op_params.filter_width = kernelWidth;                 \
  op_params.padding_values.height = (int8_t)paddingTop; \
  op_params.padding_values.width = (int8_t)paddingLeft; \
  op_params.float_activation_min = 0;                   \
  op_params.float_activation_max = 0;                   \
  op_params.quantized_activation_min = 0;               \
  op_params.quantized_activation_max = 0;

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
      throw std::runtime_error("training PoolLayer : Unsupported pool type");
  }
}

void PoolLayer::forwarding(bool) { cpu::ops::PoolLayer::run(); }

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
