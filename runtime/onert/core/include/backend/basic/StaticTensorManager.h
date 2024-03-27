/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_BASIC_STATICTENSOR_MANAGER_H__
#define __ONERT_BACKEND_BASIC_STATICTENSOR_MANAGER_H__

#include "backend/basic/DynamicTensorManager.h"
#include "backend/basic/MemoryManager.h"
#include "backend/basic/TensorRegistry.h"
#include "ir/OperandIndexMap.h"
#include "ir/OperandInfo.h"
#include "TensorRegistry.h"

namespace onert
{
namespace backend
{
namespace basic
{

class DynamicTensorManager;

class StaticTensorManager
{
private:
  using MemoryManager = backend::basic::MemoryManager<ir::OperandIndex>;

public:
  StaticTensorManager(const std::shared_ptr<TensorRegistry> &reg,
                      DynamicTensorManager *dynamic_tensor_manager);
  StaticTensorManager(const std::shared_ptr<TensorRegistry> &reg, const std::string planner_id,
                      DynamicTensorManager *dynamic_tensor_manager);
  virtual ~StaticTensorManager() = default;

  void allocateNonconsts(void);
  void deallocateNonconsts(void);

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &tensor_info,
                   ir::Layout backend_layout, bool as_const);

  void claimPlan(const ir::OperandIndex &ind, uint32_t size);
  void releasePlan(const ir::OperandIndex &ind);

  void iterate(const std::function<void(const ir::OperandIndex &)> &fn);

private:
  std::unique_ptr<MemoryManager> _nonconst_mgr;
  const std::shared_ptr<TensorRegistry> _tensors;
  ir::OperandIndexMap<bool> _as_constants;
  DynamicTensorManager *_dynamic_tensor_manager;
};

} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_STATICTENSOR_MANAGER_H__
