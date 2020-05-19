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

#include "../operand/Tensor.h"
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class DepthwiseConvolutionLayer : public ::onert::exec::IFunction
{
public:
  DepthwiseConvolutionLayer();

public:
  void convFloat32();

  void convQuant8();

  void configure(const ITensor *input, const ITensor *kernel,
                 const ITensor *bias, const uint32_t paddingLeft,
                 const uint32_t paddingRight, const uint32_t paddingTop,
                 const uint32_t paddingBottom, const uint32_t strideW, const uint32_t strideH,
                 const uint32_t multiplier, const ir::Activation activation,
                 ITensor *output);

  void run();
  void runSync()
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }

private:
  const ITensor *_input;
  const ITensor *_kernel;
  const ITensor *_bias;
  ITensor *_output;

  uint32_t _paddingLeft;
  uint32_t _paddingTop;
  uint32_t _paddingRight;
  uint32_t _paddingBottom;

  uint32_t _strideWidth;
  uint32_t _strideHeight;

  uint32_t _multiplier;

  ir::Activation _activation;
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_KERNEL_CPU_DEPTHWISECONVOLUTIONLAYER_H__
