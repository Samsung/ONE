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

#include "ir/train/TrainingInfo.h"

namespace onert
{
namespace ir
{
namespace train
{

TrainingInfo::ValidationResult TrainingInfo::isValid() const
{
  ValidationResult res;
  res.valid = true;

  if (_batch_size == 0)
  {
    res.valid = false;
    res.error_msg.append("batch size must not be zero\n");
  }

  if (_optimizer_info.optim_code == OptimizerCode::Undefined)
  {
    res.valid = false;
    res.error_msg.append("optimizer is undefined\n");
  }

  if (_optimizer_info.learning_rate <= 0.0f)
  {
    res.valid = false;
    res.error_msg.append("learning rate must be positive\n");
  }

  if (_loss_info.loss_code == LossCode::Undefined)
  {
    res.valid = false;
    res.error_msg.append("loss is undefined\n");
  }

  if (_loss_info.reduction_type == LossReductionType::Undefined)
  {
    res.valid = false;
    res.error_msg.append("loss reduction type is undefined\n");
  }

  // If there are invalid combination, add more condition-check here
  return res;
}

} // namespace train
} // namespace ir
} // namespace onert
