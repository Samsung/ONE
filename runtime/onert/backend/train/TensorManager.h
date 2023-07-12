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

#ifndef __ONERT_BACKEND_TRAIN_TENSOR_MANAGER_H__
#define __ONERT_BACKEND_TRAIN_TENSOR_MANAGER_H__

#include "MemoryManager.h"
#include "TensorRegistry.h"

#include <ir/OperandIndexMap.h>
#include <ir/OperandInfo.h>

namespace onert
{
namespace backend
{
namespace train
{

class TensorManager
{
public:
  TensorManager(const std::shared_ptr<TensorRegistry> &reg, const std::string planner_id);
  virtual ~TensorManager() = default;

  void allocateNonconsts();
  void allocateTrainableTensors();
  void deallocateNonconsts();

  // NOTE For now, whether or not to build operands to trainable tensor depends on whether
  //      the corresponding operand is constant.
  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &tensor_info,
                   ir::Layout backend_layout, bool as_const);

  void claimPlan(const ir::OperandIndex &ind, uint32_t size);
  void releasePlan(const ir::OperandIndex &ind);

  void iterate(const std::function<void(const ir::OperandIndex &)> &fn);

private:
  std::unique_ptr<MemoryManager> _nonconst_mgr;
  std::unique_ptr<MemoryManager> _trainable_mgr;
  const std::shared_ptr<TensorRegistry> _tensors;
  ir::OperandIndexMap<bool> _as_constants;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_TENSOR_MANAGER_H__
