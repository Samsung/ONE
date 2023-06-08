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
#include "ir/operation/Loss.h"

#include <bitset>

namespace onert
{
namespace ir
{
namespace train
{

struct LossInfo
{
  // NOTE How to store the type of loss here has not been decided yet. You can change it if you want
  //      it better.
  ir::operation::Loss::Type loss_type;
  //
  const void *y_true_buf{nullptr};
  ir::OperandIndex y_pred_index{};
};

class TrainingInfo
{
private:
  enum info {
    BATCH_SIZE,
    LOSS,
    END,
  };

public:
  TrainingInfo() {}
  TrainingInfo(const TrainingInfo &obj) = default;
  TrainingInfo(TrainingInfo &&) = default;
  TrainingInfo &operator=(const TrainingInfo &) = default;
  TrainingInfo &operator=(TrainingInfo &&) = default;
  ~TrainingInfo() = default;

  bool shouldTrain() const { return _has_info.all(); }
  const LossInfo &lossInfo() const { return _loss_info; }
  void setLossInfo(const LossInfo &loss_info)
  {
    _has_info.set(LOSS);
    _loss_info = loss_info;
  }
  int32_t batchsize() const { return _batchsize; }
  void setBatchSize(int32_t batchsize)
  {
    if (batchsize == 0) {
      _has_info.reset(BATCH_SIZE);
    } else {
      _has_info.set(BATCH_SIZE);
    }
    _batchsize = batchsize;
  }

private:
  std::bitset<END> _has_info;
  int32_t _batchsize;
  LossInfo _loss_info;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_TRAINING_INFO_H__
