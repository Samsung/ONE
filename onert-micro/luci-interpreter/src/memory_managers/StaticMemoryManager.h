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

#ifdef USE_STATIC_ALLOC

#ifndef LUCI_INTERPRETER_STATIC_MEMORY_MANAGER_H
#define LUCI_INTERPRETER_STATIC_MEMORY_MANAGER_H

#include "luci_interpreter/core/DataType.h"
#include "luci_interpreter/core/Tensor.h"

#include <cassert>

namespace luci_interpreter
{

// Used for allocations in static buffer, using offsets defined in luci model.
class StaticMemoryManager
{
public:
  StaticMemoryManager() = delete;

  // To initialize static memory manager with precalculating required buffers size for input,
  // output and for intermediate computations buffers.
  // Using Static Memory Manager with common buffer for input, output, and for intermediate
  // computations
  // TODO remove this *_req_size to read it from circle file
  explicit StaticMemoryManager(int32_t input_req_size, int32_t buffer_req_size,
                               int32_t output_req_size)
    : _input_buffer_ptr(nullptr), _buffer_ptr(nullptr), _output_buffer_ptr(nullptr),
      _input_req_size(input_req_size), _buffer_req_size(buffer_req_size),
      _output_req_size(output_req_size)
  { /* Do nothing */
  }

  // To set a pointer for tensor in _buffer_ptr with right offset
  uint8_t *allocate_memory(int32_t offset);
  // To set a pointer for tensor in input_buffer with right offset
  uint8_t *allocate_memory_for_input(int32_t offset);
  // To set a pointer for tensor in output_buffer with right offset
  uint8_t *allocate_memory_for_output(int32_t offset);

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
  // To delete memory for input buffer
  void release_input_buf();
  // To delete memory for output buffer
  void release_output_buf();

private:
  // Stores a pointer to the beginning of the allocated memory buffer.
  uint8_t *_buffer_ptr;
  uint8_t *_input_buffer_ptr;
  uint8_t *_output_buffer_ptr;

  // TODO remove this fields to read it from circle file
  int32_t _input_req_size{};
  int32_t _buffer_req_size{};
  int32_t _output_req_size{};
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_STATIC_MEMORY_MANAGER_H

#endif // USE_STATIC_ALLOC
