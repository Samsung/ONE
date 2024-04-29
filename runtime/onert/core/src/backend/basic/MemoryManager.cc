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

#include <backend/basic/MemoryManager.h>

#include <cassert>

#include "MemoryPlannerFactory.h"
#include "util/ConfigSource.h"
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace basic
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

basic::IMemoryPlanner<ir::OperandIndex> *MemoryManager::createMemoryPlanner()
{
  auto planner_id = util::getConfigString(util::config::CPU_MEMORY_PLANNER);
  return basic::MemoryPlannerFactory::get().create(planner_id);
}

basic::IMemoryPlanner<ir::OperandIndex> *
MemoryManager::createMemoryPlanner(const std::string planner_id)
{
  return basic::MemoryPlannerFactory::get().create(planner_id);
}

void MemoryManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  _mem_planner->claim(ind, size);
}

void MemoryManager::releasePlan(const ir::OperandIndex &ind) { _mem_planner->release(ind); }

void MemoryManager::allocate(void)
{
  _mem_alloc = std::make_shared<basic::Allocator>(_mem_planner->capacity());
  assert(_mem_alloc->base());
}

uint8_t *MemoryManager::getBuffer(const ir::OperandIndex &ind) const
{
  assert(_mem_planner->memory_plans().find(ind) != _mem_planner->memory_plans().end());
  const auto &mem_blk = _mem_planner->memory_plans().at(ind);
  return _mem_alloc->base() + mem_blk.offset;
}

std::shared_ptr<basic::Allocator> DynamicMemoryManager::allocate(const ITensor *tensor,
                                                                 uint32_t capacity)
{
  auto find = _mem_alloc_map.find(tensor);
  if (find != _mem_alloc_map.end())
    throw std::runtime_error("Cannot allocate memory for a tensor. It was already allocated.");

  _mem_alloc_map[tensor] = std::make_shared<basic::Allocator>(capacity);
  return _mem_alloc_map[tensor];
}

void DynamicMemoryManager::deallocate(const ITensor *tensor)
{
  auto find = _mem_alloc_map.find(tensor);
  if (find == _mem_alloc_map.end())
    throw std::runtime_error("Cannot find Allocator for the requested index");

  find->second->release();    // explicitly erase memory
  _mem_alloc_map.erase(find); // remove tensor and alloc
}

void DynamicMemoryManager::deallocate(void)
{
  for (auto &&mem_alloc : _mem_alloc_map)
  {
    // Release memory buffer of mem_alloc
    mem_alloc.second->release();
  }

  _mem_alloc_map.clear();
}

} // namespace basic
} // namespace backend
} // namespace onert
