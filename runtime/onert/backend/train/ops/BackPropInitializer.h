/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_OPS_BACKPROP_INITIALIZER_H__
#define __ONERT_BACKEND_TRAIN_OPS_BACKPROP_INITIALIZER_H__

#include <backend/IPortableTensor.h>
#include <cker/Types.h>
#include <exec/train/ITrainableFunction.h>

#include "../Tensor.h"

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

// TODO Introduce IFunction for only backwarding
class BackPropInitializer : public exec::train::ITrainableFunction
{
public:
  BackPropInitializer(const std::vector<BackPropTensor *> back_props);

public:
  void forward(bool training) override;
  void backward() override;

private:
  const std::vector<BackPropTensor *> _back_props;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_BACKPROP_INITIALIZER_H__
