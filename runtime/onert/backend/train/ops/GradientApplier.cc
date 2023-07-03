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

#include "GradientApplier.h"

#include <exec/train/optimizer/OptimizerContext.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

GradientApplier::GradientApplier() : _optimizer{nullptr}, _gradient_tensor{}, _trainable_tensor{}
{
  // DO NOTHING
}

void GradientApplier::configure(std::shared_ptr<exec::train::optimizer::Optimizer> optimizer,
                                const IPortableTensor *grad, ITrainableTensor *trainable)
{
  _optimizer = optimizer;
  _gradient_tensor = grad;
  _trainable_tensor = trainable;
}

void GradientApplier::backward()
{
  // TODO Apply the correct iteration
  _optimizer->applyGradient(std::forward_as_tuple(*_gradient_tensor, *_trainable_tensor, 0));
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
