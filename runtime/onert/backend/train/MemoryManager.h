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

#ifndef __ONERT_BACKEND_TRAIN_MEMORY_MANAGER_H__
#define __ONERT_BACKEND_TRAIN_MEMORY_MANAGER_H__

#include <backend/basic/MemoryManager.h>

#include "ExtraTensorIndex.h"
#include "DisposableTensorIndex.h"

#include "MemoryPlannerFactory.h"
#include <util/ConfigSource.h>
#include <cassert>

namespace onert
{
namespace backend
{
namespace train
{

using MemoryManager = backend::basic::MemoryManager;

class GradientMemoryManager : public MemoryManager
{
public:
  GradientMemoryManager(const std::string planner_id, uint32_t optimizer_vars_count);
  virtual ~GradientMemoryManager() = default;

  void allocate(void);
  uint8_t *getOptVarBuffer(const ir::OperandIndex &ind, uint32_t pos_var) const;

private:
  std::shared_ptr<basic::Allocator> _var_mem_alloc;
  uint32_t _optim_vars_count;
};

template <typename Index> class TrainMemoryManager
{
public:
  TrainMemoryManager();
  TrainMemoryManager(const std::string planner_id);

  void allocate(void);
  uint8_t *getBuffer(const Index &ind) const;
  void deallocate(void) { _mem_alloc->release(); }

  void claimPlan(const Index &ind, uint32_t size);
  void releasePlan(const Index &ind);

  std::shared_ptr<basic::Allocator> getMemAlloc() { return _mem_alloc; }

private:
  basic::IMemoryPlanner<Index> *createMemoryPlanner();
  basic::IMemoryPlanner<Index> *createMemoryPlanner(const std::string planner_id);

private:
  std::shared_ptr<basic::IMemoryPlanner<Index>> _mem_planner;
  std::shared_ptr<basic::Allocator> _mem_alloc;
};

using DisposableMemoryManager = TrainMemoryManager<DisposableTensorIndex>;
using ExtraMemoryManager = TrainMemoryManager<ExtraTensorIndex>;

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_MEMORY_MANAGER_H__
