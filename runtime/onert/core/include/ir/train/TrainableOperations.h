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

#ifndef __ONERT_IR_TRAIN_TRAINABLE_OPERATIONS_H__
#define __ONERT_IR_TRAIN_TRAINABLE_OPERATIONS_H__

#include "ir/Index.h"
#include "ir/train/ITrainableOperation.h"
#include "util/ObjectManager.h"

namespace onert
{
namespace ir
{
namespace train
{

class TrainableOperations : public util::ObjectManager<OperationIndex, ITrainableOperation>
{
public:
  TrainableOperations() = default;
  TrainableOperations(const TrainableOperations &obj);
  TrainableOperations(TrainableOperations &&) = default;
  TrainableOperations &operator=(const TrainableOperations &) = delete;
  TrainableOperations &operator=(TrainableOperations &&) = default;
  ~TrainableOperations() = default;
};

} // namespace train
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_TRAIN_TRAINABLE_OPERATIONS_H__
