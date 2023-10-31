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

#ifndef __ONERT_COMPILER_TRAIN_TRAINING_INFO_H__
#define __ONERT_COMPILER_TRAIN_TRAINING_INFO_H__

#include "ir/Index.h"
#include "ir/operation/Loss.h"
#include "ir/train/OptimizerCode.h"
#include "ir/train/OptimizerInfo.h"

namespace onert
{
namespace compiler
{
namespace train
{

struct LossInfo
{
  ir::operation::Loss::Type type;
  // TODO Add members for loss
};

class TrainingInfo
{
public:
  TrainingInfo() {}
  TrainingInfo(const TrainingInfo &obj) = default;
  TrainingInfo(TrainingInfo &&) = default;
  TrainingInfo &operator=(const TrainingInfo &) = default;
  TrainingInfo &operator=(TrainingInfo &&) = default;
  ~TrainingInfo() = default;

  uint32_t batchSize() const { return _batch_size; }
  void setBatchSize(const uint32_t batch_size) { _batch_size = batch_size; }
  const LossInfo &lossInfo() const { return _loss_info; }
  void setLossInfo(const LossInfo &loss_info) { _loss_info = loss_info; }
  const ir::train::OptimizerInfo &optimizerInfo() const { return _optimizer_info; }
  void setOptimizerInfo(const ir::train::OptimizerInfo &optimizer_info)
  {
    _optimizer_info = optimizer_info;
  }

private:
  LossInfo _loss_info{ir::operation::Loss::Type::MEAN_SQUARED_ERROR};
  ir::train::OptimizerInfo _optimizer_info{ir::train::OptimizerCode::Invalid, 0};
  uint32_t _batch_size = 0;
};

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_TRAINING_INFO_H__
