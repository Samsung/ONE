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

#include "MemoryManager.h"

#include <cassert>

namespace luci_interpreter
{

// Used for allocations in static buffer, using offsets defined in luci model.
class StaticMemoryManager : public IMemoryManager
{
public:
  StaticMemoryManager() = delete;

  // To initialize static memory manager with preallocated buffer.
  // Using Static Memory Manager with common buffer for input, output, and for intermediate
  // computations tensors.
  explicit StaticMemoryManager(uint8_t *buffer_ptr)
    : _buffer_ptr(buffer_ptr), _input_buffer_ptr(nullptr), _output_buffer_ptr(nullptr),
      _is_owning_buffers(false)
  { /* Do nothing */
  }

  // To initialize static memory manager with precalculating required buffers size for input,
  // output and for intermediate computations buffers.
  // Using Static Memory Manager with common buffer for input, output, and for intermediate
  // computations
  // TODO remove this *_req_size to reade it from circle file
  explicit StaticMemoryManager(int32_t input_req_size, int32_t buffer_req_size,
                               int32_t output_req_size)
    : _input_buffer_ptr(nullptr), _buffer_ptr(nullptr), _output_buffer_ptr(nullptr),
      _input_req_size(input_req_size), _buffer_req_size(buffer_req_size),
      _output_req_size(output_req_size), _is_owning_buffers(true)
  { /* Do nothing */
  }

  // To set a pointer for tensor in _buffer_ptr with right offset
  void allocate_memory(luci_interpreter::Tensor &tensor) final;
  // To set tensor data pointer to nullptr
  void release_memory(luci_interpreter::Tensor &tensor) final;
  // Help function to identify is static memory manager
  bool is_static_manager() const final;

  // Methods for static memory managers with split buffers (input, output and for intermediate
  // calculations)

  // To identify memory managers with split buffers
  bool is_owning_buffers() const { return _is_owning_buffers; }

  // To set a pointer for tensor in input_buffer with right offset
  void allocate_memory_for_input(luci_interpreter::Tensor &tensor);
  // To set a pointer for tensor in output_buffer with right offset
  void allocate_memory_for_output(luci_interpreter::Tensor &tensor);

  // Methods to set data pointer for tensor
  // To allocate input memory buffer with _input_req_size * size_type bytes. Result pointer -
  // _input_buffer_ptr
  void allocate_input_buf();
  // To allocate input memory buffer with _output_req_size * size_type bytes. Result pointer -
  // _output_buffer_ptr
  void allocate_output_buf();
  // To allocate intermediate computing memory buffer with _buffer_req_size * size_type bytes.
  // Result pointer - _buffer_ptr
  void allocate_computing_buf();

  // To delete memory for intermediate computing buffer
  void release_computing_buf();

private:
  void base_allocate_memory(luci_interpreter::Tensor &tensor, uint8_t *buffer_ptr);

  // Stores a pointer to the beginning of the allocated memory buffer.
  uint8_t *_buffer_ptr;
  uint8_t *_input_buffer_ptr;
  uint8_t *_output_buffer_ptr;

  // TODO remove this fields an read it from circle file
  int32_t _input_req_size{};
  int32_t _buffer_req_size{};
  int32_t _output_req_size{};

  bool _is_owning_buffers;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_STATIC_MEMORY_MANAGER_H
