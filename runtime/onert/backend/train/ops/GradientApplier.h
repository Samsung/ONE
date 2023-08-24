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

#ifndef __ONERT_BACKEND_TRAIN_OPS_GRADIENT_APPLIER_H__
#define __ONERT_BACKEND_TRAIN_OPS_GRADIENT_APPLIER_H__

#include <exec/train/IGradientApplier.h>

#include <exec/train/optimizer/Optimizer.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

class GradientApplier : public ::onert::exec::train::IGradientApplier
{
public:
  GradientApplier();
  ~GradientApplier() = default;

  void configure(const exec::train::optimizer::Optimizer *optimizer,
                 const IPortableTensor *gradient, ITrainableTensor *trainable);
  void applyGradient(uint32_t training_step) override;

private:
  const exec::train::optimizer::Optimizer *_optimizer;
  const IPortableTensor *_gradient_tensor;
  ITrainableTensor *_trainable_tensor;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_GRADIENT_APPLIER_H__
