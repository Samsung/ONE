/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file        MemoryPlanner.h
 * @brief       This file contains Memory Planning related classes
 */

#ifndef __ONERT_BACKEND_BASIC_MEMORY_PLANNER_H__
#define __ONERT_BACKEND_BASIC_MEMORY_PLANNER_H__

#include <cassert>
#include <map>
#include <vector>
#include <unordered_set>
#include <memory>

#include "backend/basic/Allocator.h"
#include "backend/basic/IMemoryPlanner.h"
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace basic
{

/**
 * @brief Class to plan memory by bump way
 */
template <typename Index> class BumpPlanner : public IMemoryPlanner<Index>
{
public:
  /**
   * @brief Claim memory for tensor by bump way
   * @param[in] index The tensor index
   * @param[in] size The size of the memory
   */
  void claim(const Index &, size_t) override;
  /**
   * @brief Release memory for tensor by bump way
   * @param[in] index The tensor index
   */
  void release(const Index &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  std::unordered_map<Index, Block> &memory_plans() override { return _mem_plans; }

private:
  uint32_t _capacity = 0;
  std::unordered_map<Index, Block> _mem_plans;
};

template <typename Index> inline void BumpPlanner<Index>::claim(const Index &ind, size_t size)
{
  Block blk{_capacity, size};
  _mem_plans[ind] = blk;
  _capacity += size;

  VERBOSE(BP_PLANNER) << "CLAIM(" << ind << "): " << blk.offset << ", " << blk.size << std::endl;
}

template <typename Index> inline void BumpPlanner<Index>::release(const Index &ind)
{
  VERBOSE(BP_PLANNER) << "RELEASE(" << ind << "): "
                      << "NOTHING does" << std::endl;
}

/**
 * @brief Class to plan memory by firstfit way
 */
template <typename Index> class FirstFitPlanner : public IMemoryPlanner<Index>
{
public:
  /**
   * @brief Claim memory for tensor by firstfit way
   * @param[in] index The tensor index
   * @param[in] size The size of the memory
   */
  void claim(const Index &, size_t) override;
  /**
   * @brief Release memory for tensor by firstfit way
   * @param[in] index The tensor index
   */
  void release(const Index &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  std::unordered_map<Index, Block> &memory_plans() override { return _mem_plans; }

private:
  uint32_t _capacity = 0;
  std::unordered_map<Index, Block> _mem_plans;
  // Use std::map because claim() assumes that _claim_table is sorted by uint32_t(base_offset)
  std::map<uint32_t, Index> _claim_table;
};

// There are some assumptions for claiming memory(== making a reservation for memory).
// 1. About _claim_table(std::map).
//   - The table's data structure is std::map so that it always sorts
//     value(Index) by key(base_offset).
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
template <typename Index> inline void FirstFitPlanner<Index>::claim(const Index &ind, size_t size)
{
  // Find the right position for claiming
  uint32_t next_offset = 0;
  for (const auto &mem_claim : _claim_table)
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
  // std::map's operator[] requires default constructor of value
  // _claim_table[next_offset] = ind;
  _claim_table.emplace(std::make_pair(next_offset, ind));
  _mem_plans[ind] = {next_offset, size};

  VERBOSE(FF_PLANNER) << "claim(" << ind << "): [+" << next_offset << ", " << size << "sz]"
                      << std::endl;

  if (_capacity < next_offset + size)
  {
    _capacity = next_offset + size;
  }
}

template <typename Index> inline void FirstFitPlanner<Index>::release(const Index &ind)
{
  for (auto it = _claim_table.cbegin(); it != _claim_table.cend(); ++it)
  {
    if (it->second == ind)
    {
      uint32_t offset = it->first;
      uint32_t size = _mem_plans[ind].size;

      _claim_table.erase(it);

      VERBOSE(FF_PLANNER) << "release(" << ind << "): [+" << offset << ", " << size << "sz]"
                          << std::endl;
      return;
    }
  }
  assert(!"Cannot release for given index. It has been not claimed or released already.");
}

/**
 * @brief Class to plan memory by Weighted Interval Color algorithm
 */
template <typename Index> class WICPlanner : public IMemoryPlanner<Index>
{
public:
  WICPlanner();

  /**
   * @brief Claim memory for tensor by WIC algorithm
   * @param[in] index The tensor index
   * @param[in] size The size of the memory
   */
  void claim(const Index &, size_t) override;
  /**
   * @brief Release memory for tensor by WIC algorithm
   * @param[in] index The tensor index
   */
  void release(const Index &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override
  {
    if (!_initialized)
      buildMemoryPlans();
    return _capacity;
  }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  std::unordered_map<Index, Block> &memory_plans() override;

private:
  void buildMemoryPlans();

  bool _initialized;
  uint32_t _capacity;
  std::unordered_map<Index, Block> _mem_plans;
  std::unordered_set<Index> _live_indices;
  std::unordered_map<Index, std::vector<Index>> _interference_graph;
  // Sort indices by descending order of size
  std::multimap<uint32_t, Index, std::greater<uint32_t>> _indices;
};

template <typename Index>
inline WICPlanner<Index>::WICPlanner()
  : _initialized(false), _capacity(0), _mem_plans(), _live_indices(), _interference_graph(),
    _indices()
{
  // DO NOTHING
}

template <typename Index> inline void WICPlanner<Index>::claim(const Index &ind, size_t size)
{
  _indices.emplace(size, ind);
  _interference_graph[ind].insert(_interference_graph[ind].end(), _live_indices.cbegin(),
                                  _live_indices.cend());
  for (const auto &live_operand : _live_indices)
  {
    _interference_graph[live_operand].emplace_back(ind);
  }
  _live_indices.emplace(ind);

  VERBOSE(WIC_PLANNER) << "claim(" << ind << "): [" << size << "sz]" << std::endl;
}

template <typename Index> inline void WICPlanner<Index>::release(const Index &ind)
{
  _live_indices.erase(ind);
  VERBOSE(WIC_PLANNER) << "release(" << ind << ")" << std::endl;
}

/*
 * Build memory plans using liveness and size of operands
 * 1. Build inference graph at claim
 *   - Two operands interfere if they have overlapped live range
 * 2. Sort operands in descending order of size
 *   - Use std::multimap to sort operands
 * 3. Allocate memory block for sorted operands
 *   - Find free memory block which does not overlap with interfered operands
 */
template <typename Index> inline void WICPlanner<Index>::buildMemoryPlans()
{
  for (const auto &operand : _indices)
  {
    uint32_t size = operand.first;
    const Index &ind = operand.second;
    VERBOSE(WIC_PLANNER) << "build_plan(" << ind << "): [" << size << "sz]" << std::endl;

    uint32_t next_offset = 0;
    if (_interference_graph.count(ind))
    {
      // Find interfered memory plans and sort them by offset
      std::multimap<uint32_t, uint32_t> interfered_plans;
      for (const auto &interference : _interference_graph[ind])
      {
        if (_mem_plans.count(interference))
          interfered_plans.emplace(_mem_plans[interference].offset, _mem_plans[interference].size);
      }

      // Find free memory block in first-fit manner
      for (const auto &interfered_plan : interfered_plans)
      {
        auto claimed_base_offset = interfered_plan.first;
        auto claimed_size = interfered_plan.second;
        VERBOSE(WIC_PLANNER) << "interfere : [+" << claimed_base_offset << ", " << claimed_size
                             << "sz]" << std::endl;
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
    else
    {
      VERBOSE(WIC_PLANNER) << "No interference" << std::endl;
    }

    _mem_plans[ind] = {next_offset, size};
    VERBOSE(WIC_PLANNER) << "alloc(" << ind << "): [+" << next_offset << ", " << size << "sz]"
                         << std::endl;

    if (_capacity < next_offset + size)
    {
      _capacity = next_offset + size;
    }
  }
  _initialized = true;
  _interference_graph.clear();
  _indices.clear();
}

template <typename Index> inline std::unordered_map<Index, Block> &WICPlanner<Index>::memory_plans()
{
  if (!_initialized)
    buildMemoryPlans();
  return _mem_plans;
}

} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_MEMORY_PLANNER_H__
