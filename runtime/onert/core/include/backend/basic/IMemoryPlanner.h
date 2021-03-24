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

#ifndef __ONERT_BACKEND_IMEMORY_PLANNER_H__
#define __ONERT_BACKEND_IMEMORY_PLANNER_H__

#include "ir/OperandIndexMap.h"

namespace onert
{
namespace backend
{
namespace basic
{

/**
 * @brief Structure to have memory offset and size
 */
struct Block
{
  uint32_t offset;
  size_t size;
};

/**
 * @brief Interface to plan memory
 */
struct IMemoryPlanner
{
  using MemoryPlans = ir::OperandIndexMap<Block>;

  /**
   * @brief Claim memory for operand
   * @param[in] index The operand index
   * @param[in] size The size of the memory
   */
  virtual void claim(const ir::OperandIndex &, size_t) = 0;
  /**
   * @brief Release memory for operand
   * @param[in] index The operand index
   */
  virtual void release(const ir::OperandIndex &) = 0;
  /**
   * @brief Get capacity for memory planning
   * @return The value of capacity
   */
  virtual uint32_t capacity() = 0;
  /**
   * @brief Get MemoryPlans
   * @return MemoryPlans
   */
  virtual MemoryPlans &memory_plans() = 0;

  virtual ~IMemoryPlanner() = default;
};

} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_IMEMORY_PLANNER_H__
