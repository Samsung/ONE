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

#include "ConvolutionLayer.h"

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

ConvolutionLayer::ConvolutionLayer() : cpu::ops::ConvolutionLayer()
{
  // DO NOTHING
}

ConvolutionLayer::~ConvolutionLayer() = default;

void ConvolutionLayer::configure(const IPortableTensor *input, const IPortableTensor *kernel,
                                 const IPortableTensor *bias, const ir::PaddingType paddingType,
                                 const uint32_t paddingLeft, const uint32_t paddingRight,
                                 const uint32_t paddingTop, const uint32_t paddingBottom,
                                 const uint32_t strideWidth, const uint32_t strideHeight,
                                 const uint32_t dilationWidthFactor,
                                 const uint32_t dilationHeightFactor,
                                 const ir::Activation activation, IPortableTensor *output)
{
  cpu::ops::ConvolutionLayer::configure(
    input, kernel, bias, paddingType, paddingLeft, paddingRight, paddingTop, paddingBottom,
    strideWidth, strideHeight, dilationWidthFactor, dilationHeightFactor, activation, output);
}

void ConvolutionLayer::forwarding(bool) { cpu::ops::ConvolutionLayer::run(); }

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
