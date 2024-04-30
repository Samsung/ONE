/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "MemoryManager.h"

#include "MemoryPlannerFactory.h"

#include <util/ConfigSource.h>

#include <cassert>

namespace onert
{
namespace backend
{
namespace train
{

DisposableMemoryManager::DisposableMemoryManager() : _mem_planner{createMemoryPlanner()}
{
  // DO NOTHING
}

DisposableMemoryManager::DisposableMemoryManager(const std::string planner_id)
  : _mem_planner{createMemoryPlanner(planner_id)}
{
  // DO NOTHING
}

basic::IMemoryPlanner<DisposableTensorIndex> *DisposableMemoryManager::createMemoryPlanner()
{
  auto planner_id = util::getConfigString(util::config::CPU_MEMORY_PLANNER);
  return MemoryPlannerFactory::get().create(planner_id);
}

basic::IMemoryPlanner<DisposableTensorIndex> *
DisposableMemoryManager::createMemoryPlanner(const std::string planner_id)
{
  return MemoryPlannerFactory::get().create(planner_id);
}

void DisposableMemoryManager::claimPlan(const DisposableTensorIndex &ind, uint32_t size)
{
  _mem_planner->claim(ind, size);
}

void DisposableMemoryManager::releasePlan(const DisposableTensorIndex &ind)
{
  _mem_planner->release(ind);
}

void DisposableMemoryManager::allocate(void)
{
  _mem_alloc = std::make_shared<basic::Allocator>(_mem_planner->capacity());
  assert(_mem_alloc->base());
}

uint8_t *DisposableMemoryManager::getBuffer(const DisposableTensorIndex &ind) const
{
  assert(_mem_planner->memory_plans().find(ind) != _mem_planner->memory_plans().end());
  const auto &mem_blk = _mem_planner->memory_plans().at(ind);
  return _mem_alloc->base() + mem_blk.offset;
}

} // namespace train
} // namespace backend
} // namespace onert
