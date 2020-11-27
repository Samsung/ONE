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

#ifndef __ONERT_BACKEND_XNNPACK_OPS_DEPTHWISE_CONVOLUTION_LAYER_H__
#define __ONERT_BACKEND_XNNPACK_OPS_DEPTHWISE_CONVOLUTION_LAYER_H__

#include "Layer.h"

namespace onert
{
namespace backend
{
namespace xnnpack
{
namespace ops
{

class DepthwiseConvolutionLayer : public Layer
{
public:
  DepthwiseConvolutionLayer(const std::shared_ptr<ExternalContext> external_context);

public:
  void configure(const IPortableTensor *input, const IPortableTensor *kernel,
                 const IPortableTensor *bias, ir::PaddingType padding_type,
                 const uint32_t padding_left, const uint32_t padding_right,
                 const uint32_t padding_top, const uint32_t padding_bottom,
                 const uint32_t stride_width, const uint32_t stride_height,
                 const uint32_t multiplier, const uint32_t dilation_width_factor,
                 const uint32_t dilation_height_factor, const ir::Activation activation,
                 IPortableTensor *output);

  void run() override;

  bool create() override;
  bool setup() override;

private:
  const IPortableTensor *_input;
  const IPortableTensor *_kernel;
  const IPortableTensor *_bias;
  IPortableTensor *_output;

  ir::PaddingType _padding_type;
  uint32_t _padding_left;
  uint32_t _padding_top;
  uint32_t _padding_right;
  uint32_t _padding_bottom;

  uint32_t _stride_width;
  uint32_t _stride_height;
  uint32_t _multiplier;
  uint32_t _dilation_width_factor;
  uint32_t _dilation_height_factor;

  ir::Activation _activation;
};

} // namespace ops
} // namespace xnnpack
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_XNNPACK_OPS_DEPTHWISE_CONVOLUTION_LAYER_H__
