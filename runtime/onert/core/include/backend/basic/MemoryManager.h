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

namespace onert
{
namespace backend
{

class ITensor;

namespace basic
{

class MemoryManager
{
public:
  MemoryManager();
  MemoryManager(const std::string);

  virtual ~MemoryManager() = default;

  void allocate(void);
  uint8_t *getBuffer(const ir::OperandIndex &ind) const;
  void deallocate(void) { _mem_alloc->release(); }

  void claimPlan(const ir::OperandIndex &ind, uint32_t size);
  void releasePlan(const ir::OperandIndex &ind);

  std::shared_ptr<Allocator> getMemAlloc() { return _mem_alloc; }

private:
  IMemoryPlanner<ir::OperandIndex> *createMemoryPlanner();
  IMemoryPlanner<ir::OperandIndex> *createMemoryPlanner(const std::string);

private:
  ir::OperandIndexMap<Block> _tensor_mem_map;
  std::shared_ptr<IMemoryPlanner<ir::OperandIndex>> _mem_planner;
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
