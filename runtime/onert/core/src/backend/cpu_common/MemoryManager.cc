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

#include "MemoryManager.h"

#include <cassert>

#include "MemoryPlannerFactory.h"
#include "util/ConfigSource.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

MemoryManager::MemoryManager() : _mem_planner{createMemoryPlanner()}
{
  // DO NOTHING
}

MemoryManager::MemoryManager(const std::string planner_id)
    : _mem_planner{createMemoryPlanner(planner_id)}
{
  // DO NOTHING
}

cpu_common::IMemoryPlanner *MemoryManager::createMemoryPlanner()
{
  auto planner_id = util::getConfigString(util::config::CPU_MEMORY_PLANNER);
  return cpu_common::MemoryPlannerFactory::get().create(planner_id);
}

cpu_common::IMemoryPlanner *MemoryManager::createMemoryPlanner(const std::string planner_id)
{
  return cpu_common::MemoryPlannerFactory::get().create(planner_id);
}

void MemoryManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  _mem_planner->claim(ind, size);
}

void MemoryManager::releasePlan(const ir::OperandIndex &ind) { _mem_planner->release(ind); }

void MemoryManager::allocate(void)
{
  _mem_alloc = std::make_shared<cpu_common::Allocator>(_mem_planner->capacity());
  assert(_mem_alloc->base());
}

uint8_t *MemoryManager::getBuffer(const ir::OperandIndex &ind) const
{
  assert(_mem_planner->memory_plans().find(ind) != _mem_planner->memory_plans().end());
  const auto &mem_blk = _mem_planner->memory_plans().at(ind);
  return _mem_alloc->base() + mem_blk.offset;
}

std::shared_ptr<cpu_common::Allocator> DynamicMemoryManager::allocate(const ir::OperandIndex &ind,
                                                                      uint32_t capacity)
{
  auto mem_alloc = std::make_shared<cpu_common::Allocator>(capacity);
  _mem_alloc_map[ind] = mem_alloc;
  return mem_alloc;
}

void DynamicMemoryManager::deallocate(const ir::OperandIndex &ind)
{
  auto find = _mem_alloc_map.find(ind);
  if (find == _mem_alloc_map.end())
    throw std::runtime_error("Cannot find Allocator for the requested index");

  // alloc's count decreases
  auto &alloc = find->second;
  alloc.reset();
}

void DynamicMemoryManager::deallocate(void)
{
  for (auto &mem_alloc : _mem_alloc_map)
  {
    // Release memory buffer of mem_alloc
    mem_alloc.second->release();
  }
}

} // namespace cpu_common
} // namespace backend
} // namespace onert
