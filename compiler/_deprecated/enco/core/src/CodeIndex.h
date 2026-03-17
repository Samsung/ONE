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

#ifndef __CODE_INDEX_H__
#define __CODE_INDEX_H__

#include <coco/IR/Block.h>
#include <coco/IR/Instr.h>

/**
 * @brief A CodeIndex denotes the index of instruction inside the whole module
 */
class CodeIndex
{
public:
  CodeIndex() = default;

public:
  CodeIndex(const coco::BlockIndex &blk_ind, const coco::InstrIndex &ins_ind)
    : _blk_ind{blk_ind}, _ins_ind{ins_ind}
  {
  }

public:
  const coco::BlockIndex &block(void) const { return _blk_ind; }
  const coco::InstrIndex &instr(void) const { return _ins_ind; }

private:
  coco::BlockIndex _blk_ind;
  coco::InstrIndex _ins_ind;
};

static inline coco::BlockIndex block_index(const coco::Block *blk)
{
  if (blk == nullptr)
  {
    return coco::BlockIndex{};
  }

  return blk->index();
}

static inline CodeIndex code_index(const coco::Instr *ins)
{
  return CodeIndex{block_index(ins->parent()), ins->index()};
}

static inline bool operator<(const CodeIndex &lhs, const CodeIndex &rhs)
{
  if (lhs.block() < rhs.block())
  {
    return true;
  }

  if (lhs.block().value() > rhs.block().value())
  {
    return false;
  }

  return lhs.instr() < rhs.instr();
}

#endif // __CODE_INDEX_H__
