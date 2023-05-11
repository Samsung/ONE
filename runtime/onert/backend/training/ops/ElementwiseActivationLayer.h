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

#ifndef __ONERT_BACKEND_TRAINING_OPS_ELEMENTWISEACTIVATIONLAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_ELEMENTWISEACTIVATIONLAYER_H__

#include <ops/ElementwiseActivationLayer.h> // From cpu backend

#include <exec/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

enum class ElementwiseActivationType
{
  kReLU
};

class ElementwiseActivationLayer : public exec::ITrainableFunction,
                                   public cpu::ops::ElementwiseActivationLayer
{
public:
  ElementwiseActivationLayer();

public:
  void configure(const IPortableTensor *input, IPortableTensor *output, float alpha, float beta,
                 ElementwiseActivationType op_type);

  void forwarding(bool training) override;

  // TODO Add required members for backwarding
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_ELEMENTWISEACTIVATIONLAYER_H__
