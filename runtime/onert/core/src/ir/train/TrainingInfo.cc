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

std::unique_ptr<TrainingInfo> TrainingInfo::createFromDefault()
{
  auto info = std::make_unique<TrainingInfo>();

  info->setBatchSize(1);

  onert::ir::train::LossInfo loss_info;
  {
    loss_info.loss_code = ir::train::LossCode::MeanSquaredError;
    loss_info.reduction_type = ir::train::LossReductionType::SumOverBatchSize;
  }
  info->setLossInfo(loss_info);

  onert::ir::train::OptimizerInfo opt_info;
  {
    opt_info.optim_code = ir::train::OptimizerCode::SGD;
    opt_info.learning_rate = 0.001f;
  }
  info->setOptimizerInfo(opt_info);

  return info;
}

bool TrainingInfo::isValid() const
{
  if (_batch_size == 0)
    return false;

  if (_optimizer_info.optim_code == onert::ir::train::OptimizerCode::Invalid)
    return false;

  if (_optimizer_info.learning_rate == 0.0f)
    return false;

  if (_loss_info.loss_code == onert::ir::train::LossCode::Invalid)
    return false;

  if (_loss_info.reduction_type == onert::ir::train::LossReductionType::Invalid)
    return false;

  // If there are invalid combination, add more condition-check here
  return true;
}

std::unique_ptr<TrainingInfo> TrainingInfo::clone() const
{
  return std::make_unique<TrainingInfo>(*this);
}

} // namespace train
} // namespace ir
} // namespace onert
