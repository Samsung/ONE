
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

#ifndef __ONERT_IR_TRAIN_TRAINING_INFO_H__
#define __ONERT_IR_TRAIN_TRAINING_INFO_H__

#include "ir/Index.h"
#include "ir/train/OptimizerCode.h"
#include "ir/train/OptimizerInfo.h"
#include "ir/train/LossInfo.h"

#include <unordered_set>

namespace onert
{
namespace ir
{
namespace train
{

class TrainingInfo final
{
public:
  TrainingInfo()
    : _loss_info(), _optimizer_info(), _batch_size(0), _training_step{0}, _trainable_ops{}
  {
  }
  TrainingInfo(const TrainingInfo &) = default;
  TrainingInfo(TrainingInfo &&) = default;
  TrainingInfo &operator=(const TrainingInfo &) = default;
  TrainingInfo &operator=(TrainingInfo &&) = default;
  ~TrainingInfo() = default;

  // getter
  const LossInfo &lossInfo() const { return _loss_info; }
  const OptimizerInfo &optimizerInfo() const { return _optimizer_info; }
  uint32_t batchSize() const { return _batch_size; }
  const uint32_t &trainingStep() const { return _training_step; }
  const std::unordered_set<OperationIndex> &getTrainableOps() const { return _trainable_ops; }

  // setter
  void setBatchSize(const uint32_t batch_size) { _batch_size = batch_size; }
  void setLossInfo(const LossInfo &loss_info) { _loss_info = loss_info; }
  void setOptimizerInfo(const OptimizerInfo &optimizer_info) { _optimizer_info = optimizer_info; }
  uint32_t &trainingStep() { return _training_step; }
  void setTrainableOps(const std::unordered_set<OperationIndex> &trainable_ops)
  {
    _trainable_ops = trainable_ops;
  }

  bool isValid() const;

private:
  LossInfo _loss_info;
  OptimizerInfo _optimizer_info;
  uint32_t _batch_size;
  uint32_t _training_step;
  std::unordered_set<OperationIndex> _trainable_ops;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_TRAINING_INFO_H__
