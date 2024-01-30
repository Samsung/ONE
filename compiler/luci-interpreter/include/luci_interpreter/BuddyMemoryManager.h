/* Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci_interpreter/MemoryManager.h"

#ifndef LUCI_INTERPRETER_BUDDY_MEMORY_MANAGER_H
#define LUCI_INTERPRETER_BUDDY_MEMORY_MANAGER_H

namespace luci_interpreter
{

class BuddyMemoryManager : public IMemoryManager
{
public:
  BuddyMemoryManager(uint8_t *memory_start, int32_t memSize);

  void allocate_memory(luci_interpreter::Tensor &tensor) final;
  void release_memory(luci_interpreter::Tensor &tensor) final;

private:
  struct Block
  {
    Block *next_free;
    bool is_free;
    uint32_t size;
    // debug field
    Block *self;
  };

  Block *_start_block;
  int32_t _num_blocks;
  uint32_t _size;
  Block *_free_blocks[32]{};

  static int32_t lowerLog2(uint32_t val)
  {
    int32_t i = 0;
    while (val >>= 1)
      i++;

    return i;
  }

  void addToBlocks(Block *block, int32_t l)
  {
    if (!block)
      return;

    block->next_free = _free_blocks[l];
    _free_blocks[l] = block;
  }

  void removeFromBlocks(const Block *block, int32_t l)
  {
    if (!block)
      return;

    Block *tmp = _free_blocks[l];

    if (block == tmp)
    {
      _free_blocks[l] = block->next_free;
      return;
    }

    while (tmp)
    {
      if (tmp->next_free == block)
      {
        tmp->next_free = block->next_free;
        return;
      }

      tmp = tmp->next_free;
    }
  }

  void divideBlock(Block *block, int32_t l)
  {
    int32_t size = ((block->size + sizeof(Block)) / 2) - sizeof(Block);

    removeFromBlocks(block, l);

    // there is no need to add to the free_blocks list here
    block->is_free = true;
    block->size = size;
    block->self = block;

    Block *buddy;
    buddy = (Block *)((uint8_t *)block + sizeof(Block) + size);
    buddy->is_free = true;
    buddy->size = size;
    buddy->self = buddy;

    addToBlocks(buddy, l - 1);
  }

  Block *mergeBlock(Block *block)
  {
    Block *buddy;

    const int32_t l = lowerLog2(block->size + sizeof(Block));

    const int64_t address = ((uint8_t *)block - (uint8_t *)_start_block);
    buddy = (Block *)((address ^ (1LL << l)) + (uint8_t *)_start_block);

    if (!buddy->is_free || buddy->size != block->size)
      return nullptr;

    if (block > buddy)
    {
      Block *x = block;
      block = buddy;
      buddy = x;
    }

    removeFromBlocks(block, l);
    removeFromBlocks(buddy, l);

    block->size = block->size * 2 + sizeof(Block);
    block->is_free = true;
    block->self = block;

    addToBlocks(block, l + 1);

    return block;
  }
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_BUDDY_MEMORY_MANAGER_H
