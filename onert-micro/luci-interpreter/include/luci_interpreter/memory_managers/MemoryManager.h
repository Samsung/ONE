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

#ifndef LUCI_INTERPRETER_MEMORY_MANAGER_H
#define LUCI_INTERPRETER_MEMORY_MANAGER_H

#include "luci_interpreter/core/DataType.h"
#include "luci_interpreter/core/Tensor.h"

namespace luci_interpreter
{

class IMemoryManager
{
public:
  virtual void allocate_memory(luci_interpreter::Tensor &tensor) = 0;
  virtual void release_memory(luci_interpreter::Tensor &tensor) = 0;
  virtual bool is_static_manager() const = 0;

  virtual ~IMemoryManager() = default;

  bool is_allocate_input() { return _is_allocate_input; }
  void is_allocate_input(bool allocate_input) { _is_allocate_input = allocate_input; }

protected:
  bool _is_allocate_input = true;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_MEMORY_MANAGER_H
