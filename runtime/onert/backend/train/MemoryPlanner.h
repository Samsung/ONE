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

/**
 * @file        MemoryPlanner.h
 * @brief       This file contains Memory Planning related classes
 */

#ifndef __ONERT_BACKEND_TRAIN_MEMORY_PLANNER_H__
#define __ONERT_BACKEND_TRAIN_MEMORY_PLANNER_H__

#include <backend/basic/IMemoryPlanner.h>

#include "DisposableTensorIndex.h"

#include <map>
#include <vector>
#include <unordered_set>
#include <memory>

namespace onert
{
namespace backend
{
namespace train
{

/**
 * @brief Class to plan memory by bump way
 */
class BumpPlanner : public basic::IMemoryPlanner<DisposableTensorIndex>
{
public:
  /**
   * @brief Claim memory for tensor by bump way
   * @param[in] index The tensor index
   * @param[in] size The size of the memory
   */
  void claim(const DisposableTensorIndex &, size_t) override;
  /**
   * @brief Release memory for tensor by bump way
   * @param[in] index The tensor index
   */
  void release(const DisposableTensorIndex &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  std::unordered_map<DisposableTensorIndex, basic::Block> &memory_plans() override
  {
    return _mem_plans;
  }

private:
  uint32_t _capacity = 0;
  std::unordered_map<DisposableTensorIndex, basic::Block> _mem_plans;
};

/**
 * @brief Class to plan memory by firstfit way
 */
class FirstFitPlanner : public basic::IMemoryPlanner<DisposableTensorIndex>
{
public:
  /**
   * @brief Claim memory for tensor by firstfit way
   * @param[in] index The tensor index
   * @param[in] size The size of the memory
   */
  void claim(const DisposableTensorIndex &, size_t) override;
  /**
   * @brief Release memory for tensor by firstfit way
   * @param[in] index The tensor index
   */
  void release(const DisposableTensorIndex &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  std::unordered_map<DisposableTensorIndex, basic::Block> &memory_plans() override
  {
    return _mem_plans;
  }

private:
  uint32_t _capacity = 0;
  std::unordered_map<DisposableTensorIndex, basic::Block> _mem_plans;
  // Use std::map because claim() assumes that _claim_table is sorted by uint32_t(base_offset)
  std::map<uint32_t, DisposableTensorIndex> _claim_table;
};

/**
 * @brief Class to plan memory by Weighted Interval Color algorithm
 */
class WICPlanner : public basic::IMemoryPlanner<DisposableTensorIndex>
{
public:
  WICPlanner();

  /**
   * @brief Claim memory for tensor by WIC algorithm
   * @param[in] index The tensor index
   * @param[in] size The size of the memory
   */
  void claim(const DisposableTensorIndex &, size_t) override;
  /**
   * @brief Release memory for tensor by WIC algorithm
   * @param[in] index The tensor index
   */
  void release(const DisposableTensorIndex &) override;
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
  std::unordered_map<DisposableTensorIndex, basic::Block> &memory_plans() override;

private:
  void buildMemoryPlans();

  bool _initialized;
  uint32_t _capacity;
  std::unordered_map<DisposableTensorIndex, basic::Block> _mem_plans;
  std::unordered_set<DisposableTensorIndex> _live_indices;
  std::unordered_map<DisposableTensorIndex, std::vector<DisposableTensorIndex>> _interference_graph;
  // Sort indices by descending order of size
  std::multimap<uint32_t, DisposableTensorIndex, std::greater<uint32_t>> _indices;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_MEMORY_PLANNER_H__
