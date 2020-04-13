/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "coco/IR/Block.h"
#include "coco/IR/Module.h"

#include <cassert>

namespace coco
{

template <> void DLinkedList<Block, Module>::joined(Module *, Block *curr_blk)
{
  assert(!curr_blk->index().valid());
  uint32_t value = 0;

  if (auto prev_blk = curr_blk->prev())
  {
    value = prev_blk->index().value() + 1;
  }

  for (auto blk = curr_blk; blk; blk = blk->next())
  {
    blk->_index.set(value++);
  }
}

template <> void DLinkedList<Block, Module>::leaving(Module *, Block *curr_blk)
{
  assert(curr_blk->index().valid());
  uint32_t value = curr_blk->index().value();

  for (auto blk = curr_blk->next(); blk; blk = blk->next())
  {
    blk->_index.set(value++);
  }

  curr_blk->_index.reset();
}

template <> BlockList *DLinkedList<Block, Module>::head(Module *m) { return m->block(); }

} // namespace coco
