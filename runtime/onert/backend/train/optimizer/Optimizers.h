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

#ifndef __ONERT_BACKEND_TRAIN_OPTIMIZER_OPTIMIZERS_H__
#define __ONERT_BACKEND_TRAIN_OPTIMIZER_OPTIMIZERS_H__

#include "Adam.h"
#include "SGD.h"

#include <ir/train/OptimizerInfo.h>

namespace onert
{
namespace backend
{
namespace train
{

std::unique_ptr<exec::train::optimizer::Optimizer>
createOptimizer(const ir::train::OptimizerInfo &optim_info)
{
  // TODO Set properties of optimizer
  if (optim_info.optim_code == ir::train::OptimizerCode::SGD)
  {
    return std::make_unique<optimizer::SGD>(optim_info.learning_rate);
  }
  else if (optim_info.optim_code == ir::train::OptimizerCode::Adam)
  {
    return std::make_unique<optimizer::Adam>(optim_info.learning_rate);
  }
  else
    throw std::runtime_error("Invalid optimizer type, " +
                             ir::train::toString(optim_info.optim_code));
}

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPTIMIZER_OPTIMIZERS_H__
