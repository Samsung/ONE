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

#ifndef __COCO_IR_BLOCK_H__
#define __COCO_IR_BLOCK_H__

#include "coco/IR/Module.forward.h"
#include "coco/IR/Block.forward.h"
#include "coco/IR/BlockIndex.h"
#include "coco/IR/Instr.h"
#include "coco/IR/Entity.h"

#include "coco/ADT/DLinkedList.h"

namespace coco
{

using BlockList = DLinkedList<Block, Module>::Head;

/**
 * @brief A unit of (grouped) instructions
 *
 * Block allows backend to manage a set of instructions as one unit, which is useful for H/W that
 * has a restriction on code size
 */
class Block final : public DLinkedList<Block, Module>::Node, public Entity
{
public:
  friend void DLinkedList<Block, Module>::joined(Module *, Block *);
  friend void DLinkedList<Block, Module>::leaving(Module *, Block *);

public:
  Block() : _instr{this}
  {
    // DO NOTHING
  }

public:
  Block(const Block &) = delete;
  Block(Block &&) = delete;

public:
  ~Block()
  {
    if (parent())
    {
      detach();
    }
  }

public:
  InstrList *instr(void) { return &_instr; }
  const InstrList *instr(void) const { return &_instr; }

public:
  const BlockIndex &index(void) const { return _index; }

private:
  BlockIndex _index;
  DLinkedList<Instr, Block>::Head _instr;
};

} // namespace coco

#endif // __COCO_IR_BLOCK_H__
