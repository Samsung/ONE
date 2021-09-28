/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_STATIC_MEMORY_MANAGER_H
#define LUCI_INTERPRETER_STATIC_MEMORY_MANAGER_H

#include "luci_interpreter/MemoryManager.h"

namespace luci_interpreter
{

// Used for allocations in static buffer, using offsets defined in luci model.
class StaticMemoryManager : public IMemoryManager
{
public:
  StaticMemoryManager() = delete;

  explicit StaticMemoryManager(uint8_t *buffer_ptr) : _buffer_ptr(buffer_ptr)
  { /* Do nothing */
  }

  void allocate_memory(luci_interpreter::Tensor &tensor) final;
  void release_memory(luci_interpreter::Tensor &tensor) final;

private:
  // Stores a pointer to the beginning of the allocated memory buffer.
  uint8_t *_buffer_ptr;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_STATIC_MEMORY_MANAGER_H
