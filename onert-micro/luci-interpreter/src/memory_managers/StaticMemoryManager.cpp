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

#include "luci_interpreter/memory_managers/StaticMemoryManager.h"

namespace luci_interpreter
{

void StaticMemoryManager::base_allocate_memory(luci_interpreter::Tensor &tensor,
                                               uint8_t *buffer_ptr)
{
  if (buffer_ptr == nullptr)
    assert("Buffer should be allocated\n");

  if (!tensor.is_allocatable())
  {
    return;
  }

  const auto offset = tensor.get_offset();
  assert(offset >= 0);
  auto tensor_ptr = buffer_ptr + offset;
  tensor.set_data_buffer(tensor_ptr);
}

void StaticMemoryManager::allocate_memory(luci_interpreter::Tensor &tensor)
{
  base_allocate_memory(tensor, _buffer_ptr);
}

void StaticMemoryManager::allocate_memory_for_input(luci_interpreter::Tensor &tensor)
{
  base_allocate_memory(tensor, _input_buffer_ptr);
}

void StaticMemoryManager::allocate_memory_for_output(luci_interpreter::Tensor &tensor)
{
  base_allocate_memory(tensor, _output_buffer_ptr);
}

void StaticMemoryManager::release_memory(luci_interpreter::Tensor &tensor)
{
  tensor.set_data_buffer(nullptr);
}

void StaticMemoryManager::allocate_input_buf()
{
  if (not _is_allocate_input)
    return;

  assert(_input_req_size > 0);
  if (_input_buffer_ptr == nullptr)
    _input_buffer_ptr = new uint8_t[_input_req_size];
}

void StaticMemoryManager::allocate_output_buf()
{
  assert(_output_req_size > 0);
  if (_output_buffer_ptr == nullptr)
    _output_buffer_ptr = new uint8_t[_output_req_size];
}

void StaticMemoryManager::allocate_computing_buf()
{
  assert(_buffer_req_size > 0);
  if (_buffer_ptr == nullptr)
    _buffer_ptr = new uint8_t[_buffer_req_size];
}

void StaticMemoryManager::release_computing_buf()
{
  delete[] _buffer_ptr;
  _buffer_ptr = nullptr;
}

void StaticMemoryManager::release_input_buf()
{
  delete[] _input_buffer_ptr;
  _input_buffer_ptr = nullptr;
}

void StaticMemoryManager::release_output_buf()
{
  delete[] _output_buffer_ptr;
  _output_buffer_ptr = nullptr;
}

} // namespace luci_interpreter
