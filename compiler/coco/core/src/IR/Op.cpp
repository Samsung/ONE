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

#include "coco/IR/Op.h"
#include "coco/IR/Step.h"
#include "coco/IR/Part.h"

#include <pepper/assert.h>

namespace coco
{
Op::~Op()
{
  // NOTE Op SHOULD NOT be referred by an instruction to be destructed
  assert(_step == nullptr);
}

Instr *Op::parent(void) const
{
  // Get the parent instruction specified by _step for root nodes
  if (_step)
  {
    // Op SHOULD BE a root node
    assert(_part == nullptr);
    assert(_step->instr() != nullptr);
    return _step->instr();
  }

  // Get the parent instruction of its parent Op for non-root nodes
  if (_part)
  {
    assert(_part->parent() != nullptr);
    return _part->parent()->parent();
  }

  return nullptr;
}

Op *Op::up(void) const
{
  if (_part)
  {
    assert(_part->parent() != nullptr);
    return _part->parent();
  }
  return nullptr;
}

//
// UnaryOP trait
//
UnaryOp::UnaryOp() : _arg{this}
{
  // DO NOTHING
}

uint32_t UnaryOp::arity(void) const
{
  // There is only one argument
  return 1;
}

Op *UnaryOp::arg(DBGARG(uint32_t, n)) const
{
  assert(n < 1);
  return arg();
}

std::set<Object *> UnaryOp::uses(void) const
{
  std::set<Object *> res;

  if (auto ifm = arg())
  {
    for (auto obj : ifm->uses())
    {
      res.insert(obj);
    }
  }

  return res;
}

//
// BinaryOp trait
//
BinaryOp::BinaryOp() : _left{this}, _right{this}
{
  // DO NOTHING
}

uint32_t BinaryOp::arity(void) const
{
  // There are two arguments
  return 2;
}

Op *BinaryOp::arg(uint32_t n) const
{
  assert(n < arity());

  return (n == 0) ? left() : right();
}

std::set<Object *> BinaryOp::uses(void) const
{
  std::set<Object *> res;

  if (auto l = left())
  {
    for (auto obj : l->uses())
    {
      res.insert(obj);
    }
  }

  if (auto r = right())
  {
    for (auto obj : r->uses())
    {
      res.insert(obj);
    }
  }

  return res;
}

//
// Additional Helpers
//
Op *root(Op *cur)
{
  while (cur->up())
  {
    cur = cur->up();
  }
  return cur;
}

} // namespace coco
