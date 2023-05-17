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

#ifndef __ONERT_BACKEND_TRAINING_OPS_POOLLAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_POOLLAYER_H__

#include <ops/PoolLayer.h>

#include <exec/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

enum class PoolType
{
  kMax,
};

class PoolLayer : public ::onert::exec::ITrainableFunction, public cpu::ops::PoolLayer
{
public:
  PoolLayer();

public:
  void configure(const IPortableTensor *input, const uint32_t paddingLeft,
                 const uint32_t paddingRight, const uint32_t paddingTop,
                 const uint32_t paddingBottom, const uint32_t strideWidth,
                 const uint32_t strideHeight, const uint32_t kernelWidth,
                 const uint32_t kernelHeight, const ir::Activation activation,
                 IPortableTensor *output, const PoolType op_type);
  void forward(bool training) override;
  void backward() override;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_POOLLAYER_H__
