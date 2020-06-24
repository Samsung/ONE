/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_OPS_MAXPOOLLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_MAXPOOLLAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

class MaxPoolLayer : public ::onert::exec::IFunction
{
public:
  MaxPoolLayer();

public:
  void maxPoolFloat32();

  void maxPoolQuant8();

  void configure(const IPortableTensor *input, const uint32_t paddingLeft,
                 const uint32_t paddingRight, const uint32_t paddingTop,
                 const uint32_t paddingBottom, const uint32_t strideWidth,
                 const uint32_t strideHeight, const uint32_t kernelWidth,
                 const uint32_t kernelHeight, const ir::Activation activation,
                 IPortableTensor *output);

  void run();

private:
  const IPortableTensor *_input;
  IPortableTensor *_output;

  uint32_t _paddingLeft;
  uint32_t _paddingTop;
  uint32_t _paddingRight;
  uint32_t _paddingBottom;

  uint32_t _strideWidth;
  uint32_t _strideHeight;
  uint32_t _kernelWidth;
  uint32_t _kernelHeight;

  ir::Activation _activation;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_MAXPOOLLAYER_H__
