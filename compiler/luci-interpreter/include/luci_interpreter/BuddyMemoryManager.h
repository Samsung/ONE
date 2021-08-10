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
  BuddyMemoryManager(uint8_t *memory_start, int memSize);

  void allocate_memory(luci_interpreter::Tensor *tensor) final;
  void release_memory(luci_interpreter::Tensor *tensor) final;

private:
  struct Block
  {
    Block *next_free;
    Block *self;
    bool is_free;
    int size;
  };
  Block *_start_block;
  int _num_blocks;
  int _size;
  Block *_free_blocks[32];

  int powOf2(int val)
  {
    int i = 0;

    while (val >>= 1)
      i++;

    return i;
  }

  void addToList(Block *block, int l)
  {
    if (!block)
      return;

    block->next_free = _free_blocks[l];
    _free_blocks[l] = block;
  }

  void removeFromList(Block *block, int l)
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
        tmp->next_free = block->next_free;

      tmp = tmp->next_free;
    }
  }

  void heapInit(void *memPool, int memSize)
  {
    int p = powOf2(memSize);
    memSize = 1 << p;

    _start_block = (Block *)memPool;
    _start_block->size = memSize - sizeof(Block);
    _start_block->is_free = true;
    _start_block->self = _start_block;

    _num_blocks = 0;
    _size = _start_block->size;

    for (int i = 0; i < 32; i++)
      _free_blocks[i] = NULL;

    addToList(_start_block, p);
  }

  Block *divide(Block *block, int l)
  {
    int size = ((block->size + sizeof(Block)) / 2) - sizeof(Block);

    removeFromList(block, l);

    block->is_free = true;
    block->size = size;
    block->self = block;

    Block *buddy;
    buddy = (Block *)((uint8_t *)block + sizeof(Block) + size);
    buddy->is_free = true;
    buddy->size = size;
    buddy->self = buddy;

    addToList(buddy, l - 1);

    return block;
  }

  Block *findBuddy(Block *block, int l)
  {
    long addr = ((uint8_t *)block - (uint8_t *)_start_block);

    return (Block *)((addr ^= (1 << l)) + (size_t)_start_block);
  }

  Block *merge(Block *block)
  {
    Block *buddy;

    int l = powOf2(block->size + sizeof(Block));

    buddy = findBuddy(block, l);

    if (!buddy->is_free || buddy->size != block->size)
      return NULL;

    if (block > buddy)
    {
      Block *x = block;
      block = buddy;
      buddy = x;
    }

    removeFromList(block, l);
    removeFromList(buddy, l);

    block->size = block->size * 2 + sizeof(Block);
    block->is_free = true;
    block->self = block;

    addToList(block, l + 1);

    return block;
  }
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_BUDDY_MEMORY_MANAGER_H
