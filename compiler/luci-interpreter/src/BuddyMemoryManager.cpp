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

#include "luci_interpreter/BuddyMemoryManager.h"

namespace luci_interpreter
{

BuddyMemoryManager::BuddyMemoryManager(uint8_t *memory_start, int32_t memSize)
{
  int32_t p = lowerLog2(memSize);

  // We assume that the requested size of memory does not exceed 4 GB
  assert(p < 32);
  memSize = 1 << p;

  _start_block = reinterpret_cast<Block *>(memory_start);
  _start_block->size = memSize - sizeof(Block);
  _start_block->is_free = true;
  _start_block->self = _start_block;
  _num_blocks = 0;
  _size = _start_block->size;

  for (auto &_free_block : _free_blocks)
    _free_block = nullptr;

  addToBlocks(_start_block, p);
}

void BuddyMemoryManager::allocate_memory(luci_interpreter::Tensor &tensor)
{
  const size_t element_size = getDataTypeSize(tensor.element_type());
  const int32_t num_elements = tensor.shape().num_elements();
  auto size = num_elements * element_size;
  auto footprint = size + sizeof(Block);
  auto l = (footprint & (footprint - 1)) == 0
             ? lowerLog2(footprint)
             : lowerLog2(footprint) + 1; // check footprint is pow_of_2

  while (!_free_blocks[l] && l < 32)
    l++;

  assert(l < 32);

  Block *tmp;
  tmp = _free_blocks[l];
  removeFromBlocks(tmp, l);

  while ((tmp->size + sizeof(Block)) / 2 >= size + sizeof(Block))
  {
    divideBlock(tmp, l);
    l--;
  }

  tmp->is_free = false;
  tmp->self = tmp;
  _num_blocks++;

  auto *data = (uint8_t *)(tmp + 1);
  tensor.set_data_buffer(data);
}

void BuddyMemoryManager::release_memory(luci_interpreter::Tensor &tensor)
{
  auto data = tensor.data<void>();
  auto *tmp = (Block *)((uint8_t *)data - sizeof(Block));

  assert(tmp->self == tmp);

  tmp->is_free = true;
  addToBlocks(tmp, lowerLog2(tmp->size + sizeof(Block)));

  while (tmp)
    if (tmp->size == _size)
      break;
    else
      tmp = mergeBlock(tmp);

  _num_blocks--;
  tensor.set_data_buffer(nullptr);
}

} // namespace luci_interpreter
