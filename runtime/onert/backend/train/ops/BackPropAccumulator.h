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

#ifndef __ONERT_BACKEND_TRAIN_OPS_BACKPROP_ACCUMULATOR_H__
#define __ONERT_BACKEND_TRAIN_OPS_BACKPROP_ACCUMULATOR_H__

#include <backend/IPortableTensor.h>
#include <cker/Types.h>
#include <exec/train/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

// TODO Introduce IFunction for only backwarding
class BackPropAccumulator : public exec::train::ITrainableFunction
{
public:
  BackPropAccumulator(const IPortableTensor *disposable_tensor, IPortableTensor *back_prop_tensor);
  ~BackPropAccumulator() = default;

public:
  void forward(bool training) override;
  void backward() override;

private:
  const IPortableTensor *_disposable_tensor;
  IPortableTensor *_back_prop_tensor;

  nnfw::cker::BinaryArithmeticOpParam _op_params;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_BACKPROP_ACCUMULATOR_H__
