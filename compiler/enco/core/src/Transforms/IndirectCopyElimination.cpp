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

#include "IndirectCopyElimination.h"

#include <cassert>

namespace
{

coco::Copy *as_copy(coco::Instr *ins) { return ins ? ins->asCopy() : nullptr; }

/**
 * @brief Return a set of copy instructions that are accessible from top-level module
 */
std::set<coco::Copy *> linked_copy_instrs(coco::Module *m)
{
  std::set<coco::Copy *> res;

  for (uint32_t n = 0; n < m->entity()->instr()->size(); ++n)
  {
    auto ins = m->entity()->instr()->at(n);
    assert(ins != nullptr);

    if (ins->parent() && ins->parent()->parent())
    {
      if (auto copy = ins->asCopy())
      {
        res.insert(copy);
      }
    }
  }

  return res;
}

} // namespace

namespace enco
{

void eliminate_indirect_copy(enco::Code *code)
{
  auto m = code->module();

  for (auto child : linked_copy_instrs(m))
  {
    auto from = child->from();
    assert(from != nullptr);

    // Find the irreducible origin
    while (true)
    {
      if (auto producer = coco::producer(from))
      {
        if (auto parent = as_copy(producer->loc()))
        {
          assert(parent->from() != nullptr);
          from = parent->from();
          continue;
        }
      }

      break;
    }

    child->from(from);
  }
}

} // namespace enco
