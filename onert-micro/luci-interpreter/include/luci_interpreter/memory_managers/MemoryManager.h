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

  virtual ~IMemoryManager() = default;

  bool is_allocate_input() const { return _is_allocate_input; }
  void is_allocate_input(bool allocate_input) { _is_allocate_input = allocate_input; }

  // Methods for static allocations
  // Methods to set data pointer for tensor
  // To allocate input memory buffer with _input_req_size * size_type bytes. Result pointer -
  // _input_buffer_ptr
  virtual void allocate_input_buf() = 0;
  // To allocate input memory buffer with _output_req_size * size_type bytes. Result pointer -
  // _output_buffer_ptr
  virtual void allocate_output_buf() = 0;
  // To allocate intermediate computing memory buffer with _buffer_req_size * size_type bytes.
  // Result pointer - _buffer_ptr
  virtual void allocate_computing_buf() = 0;

  // To delete memory for intermediate computing buffer
  virtual void release_computing_buf() = 0;
  // To delete memory for input buffer
  virtual void release_input_buf() = 0;
  // To delete memory for output buffer
  virtual void release_output_buf() = 0;

  // To set a pointer for tensor in input_buffer with right offset
  virtual void allocate_memory_for_input(luci_interpreter::Tensor &tensor) = 0;
  // To set a pointer for tensor in output_buffer with right offset
  virtual void allocate_memory_for_output(luci_interpreter::Tensor &tensor) = 0;

protected:
  bool _is_allocate_input = true;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_MEMORY_MANAGER_H
