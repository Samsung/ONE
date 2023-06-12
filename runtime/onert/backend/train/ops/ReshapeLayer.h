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

#ifndef __ONERT_BACKEND_TRAIN_OPS_RESHAPELAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_RESHAPELAYER_H__

#include <ops/ReshapeLayer.h>

#include <exec/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

class ReshapeLayer : public ::onert::exec::ITrainableFunction, public cpu::ops::ReshapeLayer
{
public:
  ReshapeLayer();

  void configure(const IPortableTensor *input, const IPortableTensor *shape,
                 IPortableTensor *output);
  void forward(bool training) override;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_RESHAPELAYER_H__
