/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ONERT_MICRO_CORE_MEMORY_RUNTIME_ALLOCATOR_H
#define ONERT_MICRO_CORE_MEMORY_RUNTIME_ALLOCATOR_H

#include "OMStatus.h"
#include "core/OMRuntimeContext.h"
#include "core/OMRuntimeStorage.h"

#include <vector>
#include <cstdint>

namespace onert_micro
{
namespace core
{
namespace memory
{

class OMRuntimeAllocator
{
private:
  std::vector<std::vector<uint16_t>> _alloc_plan;
  std::vector<std::vector<uint16_t>> _dealloc_plan;

public:
  OMRuntimeAllocator() = default;
  OMRuntimeAllocator(const OMRuntimeAllocator &) = delete;
  OMRuntimeAllocator &operator=(const OMRuntimeAllocator &) = delete;
  OMRuntimeAllocator &&operator=(const OMRuntimeAllocator &&) = delete;
  OMRuntimeAllocator(OMRuntimeAllocator &&) = default;
  ~OMRuntimeAllocator() = default;

  void saveAllocPlan(std::vector<std::vector<uint16_t>> &&alloc_plan)
  {
    _alloc_plan.clear();
    _alloc_plan = std::move(alloc_plan);
  }

  void saveDeallocPlan(std::vector<std::vector<uint16_t>> &&dealloc_plan)
  {
    _dealloc_plan.clear();
    _dealloc_plan = std::move(dealloc_plan);
  }

  std::vector<std::vector<uint16_t>> &getAllocPlan() { return _alloc_plan; }

  std::vector<std::vector<uint16_t>> &getDeallocPlan() { return _dealloc_plan; }

  OMStatus allocateGraphInputs(OMRuntimeContext *context, OMRuntimeStorage *storage);

  OMStatus clearAllTensorsData(OMRuntimeContext *context, OMRuntimeStorage *storage);

  OMStatus allocate(size_t kernel_index, OMRuntimeContext *context, OMRuntimeStorage *storage);
  OMStatus deallocate(size_t kernel_index, OMRuntimeStorage *storage);
};

} // namespace memory
} // namespace core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_MEMORY_RUNTIME_ALLOCATOR_H
