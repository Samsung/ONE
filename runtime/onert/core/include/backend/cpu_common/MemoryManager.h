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
#include "backend/IMemoryManager.h"
#include "IMemoryPlanner.h"
#include "ir/OperandIndexMap.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

/**
 * @brief MemoryManager for static tensors whose memory is allocated at compilation phase
 */
class MemoryManager : public backend::IMemoryManager
{
public:
  MemoryManager();
  MemoryManager(const std::string);
  virtual ~MemoryManager() = default;

  void allocate(void) override;
  uint8_t *getBuffer(const ir::OperandIndex &ind) const;
  void deallocate(void) override { _mem_alloc->release(); }

  void claimPlan(const ir::OperandIndex &ind, uint32_t size);
  void releasePlan(const ir::OperandIndex &ind);

private:
  IMemoryPlanner *createMemoryPlanner();
  IMemoryPlanner *createMemoryPlanner(const std::string);

private:
  ir::OperandIndexMap<Block> _tensor_mem_map;
  std::shared_ptr<IMemoryPlanner> _mem_planner;
  std::shared_ptr<Allocator> _mem_alloc;
};

/**
 * @brief MemoryManager for constant tensors
 */
class ConstMemoryManager
{
public:
  ConstMemoryManager() = default;
  virtual ~ConstMemoryManager() = default;

  std::shared_ptr<Allocator> allocate(const ir::OperandIndex &ind, uint32_t capacity);
  void deallocate(void);

private:
  ir::OperandIndexMap<std::shared_ptr<Allocator>> _mem_alloc_map;
};

/**
 * @brief MemoryManager for dynamic tensors whose memory is allocated at execution phase
 */
class DynamicMemoryManager
{
public:
  DynamicMemoryManager() = default;
  virtual ~DynamicMemoryManager() = default;

  std::shared_ptr<Allocator> allocate(const ir::OperandIndex &ind, uint32_t capacity);
  void deallocate(const ir::OperandIndex &ind);
  void deallocate(void);

private:
  ir::OperandIndexMap<std::shared_ptr<Allocator>> _mem_alloc_map;
};

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_MEMORY_MANAGER_H__
