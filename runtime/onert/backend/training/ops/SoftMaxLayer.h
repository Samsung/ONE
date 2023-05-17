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

#ifndef __ONERT_BACKEND_TRAINING_OPS_SOFTMAXLAYER_H__
#define __ONERT_BACKEND_TRAINING_OPS_SOFTMAXLAYER_H__

#include <ops/SoftMaxLayer.h>

#include <exec/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

class SoftMaxLayer : public ::onert::exec::ITrainableFunction, public cpu::ops::SoftMaxLayer
{
public:
  SoftMaxLayer();

  void configure(const IPortableTensor *input, const float beta, IPortableTensor *output);
  void forward(bool training) override;
  void backward() override;
};

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_OPS_SOFTMAXLAYER_H__
