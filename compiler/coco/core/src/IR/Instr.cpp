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

#include "coco/IR/Instr.h"
#include "coco/IR/Block.h"

#include <cassert>

namespace coco
{

template <> void DLinkedList<Instr, Block>::joined(Block *, Instr *curr_ins)
{
  assert(!curr_ins->index().valid());
  uint32_t value = 0;

  if (auto prev_ins = curr_ins->prev())
  {
    value = prev_ins->index().value() + 1;
  }

  for (auto ins = curr_ins; ins; ins = ins->next())
  {
    ins->_index.set(value++);
  }
}

template <> void DLinkedList<Instr, Block>::leaving(Block *, Instr *curr_ins)
{
  assert(curr_ins->index().valid());
  uint32_t value = curr_ins->index().value();

  for (auto ins = curr_ins->next(); ins; ins = ins->next())
  {
    ins->_index.set(value++);
  }

  curr_ins->_index.reset();
}

template <> InstrList *DLinkedList<Instr, Block>::head(Block *b) { return b->instr(); }

} // namespace coco
