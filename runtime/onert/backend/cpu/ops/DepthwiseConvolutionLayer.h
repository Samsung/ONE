/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_KERNEL_CPU_DEPTHWISECONVOLUTIONLAYER_H__
#define __ONERT_KERNEL_CPU_DEPTHWISECONVOLUTIONLAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"
#include "../ExternalContext.h"

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

class DepthwiseConvolutionLayer : public ::onert::exec::IFunction
{
public:
  DepthwiseConvolutionLayer() = default;

public:
  void convFloat32();

  void convQ8uPerTensor();
  void convQ8uPerChannel();

  void convQ8i();
  void convQ8iHybridPerChannel();

  void configure(const IPortableTensor *input, const IPortableTensor *kernel,
                 const IPortableTensor *bias, const uint32_t paddingLeft,
                 const uint32_t paddingRight, const uint32_t paddingTop,
                 const uint32_t paddingBottom, const uint32_t strideW, const uint32_t strideH,
                 const uint32_t multiplier, const uint32_t dilationWidth,
                 const uint32_t dilationHeight, const ir::Activation activation,
                 IPortableTensor *output, const std::shared_ptr<ExternalContext> &external_context);

  void run() override;

private:
  void prepareQ8i();
  void prepareQ8uPerChannel();
  void prepareQ8iHybridPerChannel();
  void ensureQ8iHybridPerChannel();

protected:
  const IPortableTensor *_input{nullptr};
  const IPortableTensor *_kernel{nullptr};
  const IPortableTensor *_bias{nullptr};
  IPortableTensor *_output{nullptr};

  uint32_t _paddingLeft{0};
  uint32_t _paddingTop{0};
  uint32_t _paddingRight{0};
  uint32_t _paddingBottom{0};

  uint32_t _strideWidth{0};
  uint32_t _strideHeight{0};

  uint32_t _multiplier{0};

  uint32_t _dilationWidth{1};
  uint32_t _dilationHeight{1};

  ir::Activation _activation{ir::Activation::NONE};

private:
  std::shared_ptr<ExternalContext> _external_context;

  bool _prepared{false};

  // Per channel output multiplier and shift.
  std::vector<int32_t> _per_channel_output_multiplier;
  std::vector<int> _per_channel_output_shift;

  // For hybrid
  bool _is_hybrid{false};
  std::vector<int8_t> _input_quantized;
  std::vector<float> _input_scaling_factors;
  std::vector<int32_t> _input_offsets;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_KERNEL_CPU_DEPTHWISECONVOLUTIONLAYER_H__
