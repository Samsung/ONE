/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

// This file will be removed or unified with backend/cpu_common/MemoryManager.cc

#include "MemoryManager.h"

#include <cassert>

#include "util/ConfigSource.h"
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

void BumpPlanner::claim(const ir::OperandIndex &ind, size_t size)
{
  assert(size != 0);

  Block blk{_capacity, size};
  _mem_plans[ind] = blk;
  _capacity += size;

  VERBOSE(BP_PLANNER) << "CLAIM(#" << ind.value() << "): " << blk.offset << ", " << blk.size
                      << std::endl;
}

void BumpPlanner::release(const ir::OperandIndex &ind)
{
  VERBOSE(BP_PLANNER) << "RELEASE(#" << ind.value() << "): "
                      << "NOTHING does" << std::endl;
}

// There are some assumptions for claiming memory(== making a reservation for memory).
// 1. About _claim_table(std::map).
//   - The table's data structure is std::map so that it always sorts
//     value(OperandIndex) by key(base_offset).
//   - This claim() inserts key/value into _claim_table and the release() removes the key/value from
//     _claim_table.
//   - _claim_table shows the memory status at a certain point in time. Therefore,
//     - If _claim_table has an offset and a certain size at a certain point in time,
//       it means the place at the offset has been already claimed(== can't claim now. need to find
//       someplace new).
//     - If _claim_table doesn't have any element for an offset and a certain size at a certain
//       point in time, it means the place at the offset can be claimed.
// 2. In the loop for _claim_table, we can assume the current claim_base_offset value is bigger than
//    the previous claim_base_offset.
void FirstFitPlanner::claim(const ir::OperandIndex &ind, size_t size)
{
  assert(size != 0);

  // Find the right position for claiming
  uint32_t next_offset = 0;
  for (auto &mem_claim : _claim_table)
  {
    auto claimed_base_offset = mem_claim.first;
    auto claimed_size = _mem_plans[mem_claim.second].size;
    if (next_offset + size <= claimed_base_offset)
    {
      break;
    }
    else
    {
      next_offset = claimed_base_offset + claimed_size;
    }
  }

  // Now next_offset is set to the proper offset
  _claim_table[next_offset] = ind;
  _mem_plans[ind] = {next_offset, size};

  VERBOSE(FF_PLANNER) << "claim(#" << ind.value() << "): [+" << next_offset << ", " << size << "sz]"
                      << std::endl;

  if (_capacity < next_offset + size)
  {
    _capacity = next_offset + size;
  }
}

void FirstFitPlanner::release(const ir::OperandIndex &ind)
{
  for (auto it = _claim_table.cbegin(); it != _claim_table.cend(); ++it)
  {
    if (it->second == ind)
    {
      uint32_t offset = it->first;
      uint32_t index = ind.value();
      uint32_t size = _mem_plans[ind].size;

      _claim_table.erase(it);

      VERBOSE(FF_PLANNER) << "release(#" << index << "): [+" << offset << ", " << size << "sz]"
                          << std::endl;
      return;
    }
  }
  assert(!"Cannot release for given index. It has been not claimed or released already.");
}

WICPlanner::WICPlanner()
    : _initialized(false), _capacity(0), _mem_plans(), _live_operands(), _interference_graph(),
      _map_size_to_operands(), _claim_table()
{
  // DO NOTHING
}

void WICPlanner::claim(const ir::OperandIndex &ind, size_t size)
{
  assert(size != 0);

  _map_size_to_operands.insert({size, ind});
  for (auto &live_operand : _live_operands)
  {
    _interference_graph[live_operand].insert(ind);
    _interference_graph[ind].insert(live_operand);
  }
  _live_operands.insert(ind);

  VERBOSE(WIC_PLANNER) << "claim(#" << ind.value() << "): [" << size << "sz]" << std::endl;
}

void WICPlanner::release(const ir::OperandIndex &ind)
{
  _live_operands.erase(ind);
  VERBOSE(WIC_PLANNER) << "release(#" << ind.value() << ")" << std::endl;
}

/*
 * Build memory plans using liveness and size of operands
 * 1. Build inference graph at claim
 *   - Two operands interfere if they have overlapped live range
 * 2. Sort operands descending order of size
 *   - Use std::multimap to sort operands
 * 3. Allocate memory block for sorted operands
 *   - Find free memory block which does not overlap with interfered operands
 */
void WICPlanner::buildMemoryPlans()
{
  for (auto &size_to_operand : _map_size_to_operands)
  {
    uint32_t size = size_to_operand.first;
    ir::OperandIndex ind = size_to_operand.second;
    VERBOSE(WIC_PLANNER) << "build_plan(#" << ind.value() << "): [" << size << "sz]" << std::endl;

    // Find firstfit which does not interfere with live operands
    uint32_t next_offset = 0;
    if (_interference_graph.find(ind) != _interference_graph.end())
    {
      std::unordered_set<ir::OperandIndex> &interferences = _interference_graph.find(ind)->second;
      for (auto &mem_claim : _claim_table)
      {
        if (interferences.find(mem_claim.second) != interferences.end())
        {
          auto claimed_base_offset = mem_claim.first;
          auto claimed_size = _mem_plans[mem_claim.second].size;
          VERBOSE(WIC_PLANNER) << "interfere (#" << mem_claim.second.value() << "): [+"
                               << claimed_base_offset << ", " << claimed_size << "sz]" << std::endl;
          if (next_offset + size <= claimed_base_offset)
          {
            break;
          }
          else if (next_offset < claimed_base_offset + claimed_size)
          {
            next_offset = claimed_base_offset + claimed_size;
          }
        }
      }
    }
    else
    {
      VERBOSE(WIC_PLANNER) << "No interference" << std::endl;
    }

    _claim_table.insert({next_offset, ind});
    _mem_plans[ind] = {next_offset, size};
    VERBOSE(WIC_PLANNER) << "alloc(#" << ind.value() << "): [+" << next_offset << ", " << size
                         << "sz]" << std::endl;

    if (_capacity < next_offset + size)
    {
      _capacity = next_offset + size;
    }
  }
  _initialized = true;
  _interference_graph.clear();
  _map_size_to_operands.clear();
  _claim_table.clear();
}

WICPlanner::MemoryPlans &WICPlanner::memory_plans()
{
  if (!_initialized)
    buildMemoryPlans();
  return _mem_plans;
}

class MemoryPlannerFactory
{
public:
  static MemoryPlannerFactory &get();

private:
  MemoryPlannerFactory() = default;

public:
  IMemoryPlanner *create(const std::string &key);
};

MemoryPlannerFactory &MemoryPlannerFactory::get()
{
  static MemoryPlannerFactory instance;
  return instance;
}

IMemoryPlanner *MemoryPlannerFactory::create(const std::string &key)
{
  if (key == "FirstFit")
  {
    return new FirstFitPlanner;
  }
  else if (key == "Bump")
  {
    return new BumpPlanner;
  }
  else if (key == "WIC")
  {
    return new WICPlanner;
  }
  return new FirstFitPlanner; // Default Planner
}

MemoryManager::MemoryManager() : _mem_planner{createMemoryPlanner()}
{
  // DO NOTHING
}

MemoryManager::MemoryManager(const std::string planner_id)
    : _mem_planner{createMemoryPlanner(planner_id)}
{
  // DO NOTHING
}

IMemoryPlanner *MemoryManager::createMemoryPlanner()
{
  auto planner_id = util::getConfigString(util::config::CPU_MEMORY_PLANNER);
  return MemoryPlannerFactory::get().create(planner_id);
}

IMemoryPlanner *MemoryManager::createMemoryPlanner(const std::string planner_id)
{
  return MemoryPlannerFactory::get().create(planner_id);
}

void MemoryManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  _mem_planner->claim(ind, size);
}

void MemoryManager::releasePlan(const ir::OperandIndex &ind) { _mem_planner->release(ind); }

void MemoryManager::allocate(void)
{
  _mem_alloc = std::make_shared<Allocator>(_mem_planner->capacity());
  assert(_mem_alloc->base());
}

uint8_t *MemoryManager::getBuffer(const ir::OperandIndex &ind) const
{
  assert(_mem_planner->memory_plans().find(ind) != _mem_planner->memory_plans().end());
  const auto &mem_blk = _mem_planner->memory_plans().at(ind);
  return _mem_alloc->base() + mem_blk.offset;
}

std::shared_ptr<Allocator> DynamicMemoryManager::allocate(const ir::OperandIndex &ind,
                                                          uint32_t capacity)
{
  auto mem_alloc = std::make_shared<Allocator>(capacity);
  _mem_alloc_map[ind] = mem_alloc;
  return mem_alloc;
}

void DynamicMemoryManager::deallocate(void)
{
  for (auto &mem_alloc : _mem_alloc_map)
  {
    mem_alloc.second->release();
  }
}

} // namespace controlflow
} // namespace backend
} // namespace onert
