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

#include "ir/operation/Loss.h"
#include "ir/train/OptimizerCode.h"
#include "ir/train/OptimizerInfo.h"

namespace onert
{
namespace ir
{
namespace train
{

struct LossInfo
{
  ir::operation::Loss::Type type;
  // TODO Add members for loss
};

// NOTE : This code is almost same with 'compiler::train::TrainingInfo' except _epoch
/**
 * Q: What is this class for?
 * A: This class is return type of 'traininfo_loader::loadTrainInfo()'.
 *    'traininfo_loader' is resposible for parsing metadata in model and create ir::TrainingInfo.
 *
 * Q: Instead of defining new class, Can't we use 'nnfw_train_info' in
 *    onert/api/include/nnfw_experimental.h?
 * A: It's a bit a ackward the 'onert/core' use structure defined in 'onert/api'.
 *
 *
 * Q: Instead of defining new class, Can't we use 'compiler::train::TrainingInfo'?
 * A: The role of each class is different a bit.
 *    So, I understood it is better to clearify the role by using different class.
 *
 *    * compiler::train::TrainingInfo - the argument to create Compiler
 *    * ir::train::TrainingInfo - the parsing result from train_loader
 *
 * NOTE: In the view of api, the same infomation converted into other clasess by its usage.
 *         * (ir::train::TrainingInfo) -> (nnfw_train_info) -> (compiler::train::TrainingInfo)
 */
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
  void setEpoch(uint32_t epoch) { _epoch = epoch; }
  uint32_t epoch() const { return _epoch; }

private:
  LossInfo _loss_info{operation::Loss::Type::MEAN_SQUARED_ERROR};
  OptimizerInfo _optimizer_info{OptimizerCode::Invalid, 0};
  uint32_t _batch_size = 0;
  uint32_t _epoch = 0;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_TRAINING_INFO_H__
