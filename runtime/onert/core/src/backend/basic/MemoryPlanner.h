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

#include <map>
#include <vector>
#include <unordered_set>
#include <memory>

#include "backend/basic/Allocator.h"
#include "backend/basic/IMemoryPlanner.h"
#include "ir/OperandIndexMap.h"

namespace onert
{
namespace backend
{
namespace basic
{

/**
 * @brief Class to plan memory by bump way
 */
class BumpPlanner : public IMemoryPlanner<ir::OperandIndex>
{
public:
  /**
   * @brief Claim memory for operand by bump way
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  void claim(const ir::OperandIndex &, size_t) override;
  /**
   * @brief Release memory for operand by bump way
   * @param[in] index The operand index
   */
  void release(const ir::OperandIndex &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  MemoryPlans &memory_plans() override { return _mem_plans; }

private:
  uint32_t _capacity = 0;
  MemoryPlans _mem_plans;
};

/**
 * @brief Class to plan memory by firstfit way
 */
class FirstFitPlanner : public IMemoryPlanner<ir::OperandIndex>
{
public:
  /**
   * @brief Claim memory for operand by firstfit way
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  void claim(const ir::OperandIndex &, size_t) override;
  /**
   * @brief Release memory for operand by firstfit way
   * @param[in] index The operand index
   */
  void release(const ir::OperandIndex &) override;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  uint32_t capacity() override { return _capacity; }
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  MemoryPlans &memory_plans() override { return _mem_plans; }

private:
  uint32_t _capacity = 0;
  MemoryPlans _mem_plans;
  // Use std::map because claim() assumes that _claim_table is sorted by uint32_t(base_offset)
  std::map<uint32_t, ir::OperandIndex> _claim_table;
};

/**
 * @brief Class to plan memory by Weighted Interval Color algorithm
 */
class WICPlanner : public IMemoryPlanner<ir::OperandIndex>
{
public:
  WICPlanner();

  /**
   * @brief Claim memory for operand by WIC algorithm
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  void claim(const ir::OperandIndex &, size_t) override;
  /**
   * @brief Release memory for operand by WIC algorithm
   * @param[in] index The operand index
   */
  void release(const ir::OperandIndex &) override;
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
  MemoryPlans &memory_plans() override;

private:
  void buildMemoryPlans();

  bool _initialized;
  uint32_t _capacity;
  MemoryPlans _mem_plans;
  std::unordered_set<ir::OperandIndex> _live_operands;
  ir::OperandIndexMap<std::vector<ir::OperandIndex>> _interference_graph;
  // Sort operands by descending order of size
  std::multimap<uint32_t, ir::OperandIndex, std::greater<uint32_t>> _operands;
};

} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_MEMORY_PLANNER_H__
