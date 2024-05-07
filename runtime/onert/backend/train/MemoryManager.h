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

#include "DisposableTensorIndex.h"

namespace onert
{
namespace backend
{
namespace train
{

using MemoryManager = backend::basic::MemoryManager;

class DisposableMemoryManager
{
public:
  DisposableMemoryManager();
  DisposableMemoryManager(const std::string planner_id);

  void allocate(void);
  uint8_t *getBuffer(const DisposableTensorIndex &ind) const;
  void deallocate(void) { _mem_alloc->release(); }

  void claimPlan(const DisposableTensorIndex &ind, uint32_t size);
  void releasePlan(const DisposableTensorIndex &ind);

  std::shared_ptr<basic::Allocator> getMemAlloc() { return _mem_alloc; }

private:
  basic::IMemoryPlanner<DisposableTensorIndex> *createMemoryPlanner();
  basic::IMemoryPlanner<DisposableTensorIndex> *createMemoryPlanner(const std::string planner_id);

private:
  std::shared_ptr<basic::IMemoryPlanner<DisposableTensorIndex>> _mem_planner;
  std::shared_ptr<basic::Allocator> _mem_alloc;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_MEMORY_MANAGER_H__
