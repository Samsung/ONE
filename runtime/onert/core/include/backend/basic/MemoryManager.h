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

#ifndef __ONERT_BACKEND_CPU_MEMORY_MANAGER_H__
#define __ONERT_BACKEND_CPU_MEMORY_MANAGER_H__

#include "Allocator.h"
#include "IMemoryPlanner.h"
#include "MemoryPlannerFactory.h"
#include "util/logging.h"

#include <cassert>

namespace onert
{
namespace backend
{

class ITensor;

namespace basic
{

template <typename Index> class MemoryManager
{
public:
  MemoryManager() : _mem_planner{createMemoryPlanner()}
  {
    // DO NOTHING
  }

  MemoryManager(const std::string planner_id) : _mem_planner{createMemoryPlanner(planner_id)}
  {
    // DO NOTHING
  }

  virtual ~MemoryManager() = default;

  void allocate(void)
  {
    _mem_alloc = std::make_shared<basic::Allocator>(_mem_planner->capacity());
    assert(_mem_alloc->base());
  }

  uint8_t *getBuffer(const Index &ind) const
  {
    assert(_mem_planner->memory_plans().find(ind) != _mem_planner->memory_plans().end());
    const auto &mem_blk = _mem_planner->memory_plans().at(ind);
    return _mem_alloc->base() + mem_blk.offset;
  }

  void deallocate(void) { _mem_alloc->release(); }

  void claimPlan(const Index &ind, uint32_t size) { _mem_planner->claim(ind, size); }

  void releasePlan(const Index &ind) { _mem_planner->release(ind); }

  std::shared_ptr<Allocator> getMemAlloc() { return _mem_alloc; }

private:
  IMemoryPlanner<Index> *createMemoryPlanner()
  {
    auto planner_id = util::getConfigString(util::config::CPU_MEMORY_PLANNER);
    return basic::MemoryPlannerFactory::get().create<Index>(planner_id);
  }

  IMemoryPlanner<Index> *createMemoryPlanner(const std::string planner_id)
  {
    return basic::MemoryPlannerFactory::get().create<Index>(planner_id);
  }

private:
  ir::OperandIndexMap<Block> _tensor_mem_map;
  std::shared_ptr<IMemoryPlanner<Index>> _mem_planner;
  std::shared_ptr<Allocator> _mem_alloc;
};

class DynamicMemoryManager
{
public:
  DynamicMemoryManager() = default;
  virtual ~DynamicMemoryManager() = default;

  std::shared_ptr<Allocator> allocate(const ITensor *tensor, uint32_t capacity);
  void deallocate(const ITensor *tensor);
  void deallocate(void);

private:
  std::unordered_map<const ITensor *, std::shared_ptr<Allocator>> _mem_alloc_map;
};

} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_MEMORY_MANAGER_H__
