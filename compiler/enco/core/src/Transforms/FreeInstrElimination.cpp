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

#include "FreeInstrElimination.h"

#include <cassert>
#include <set>

namespace
{

/**
 * @brief Return the set of "free" instructions in a given module
 */
std::set<coco::Instr *> free_instrs(const coco::Module *m)
{
  std::set<coco::Instr *> res;

  for (uint32_t n = 0; n < m->entity()->instr()->size(); ++n)
  {
    if (auto ins = m->entity()->instr()->at(n))
    {
      if (ins->parent() == nullptr)
      {
        res.insert(ins);
      }
    }
  }

  return res;
}

void destroy(coco::Instr *ins)
{
  auto m = ins->module();
  m->entity()->instr()->destroy(ins);
}

} // namespace

namespace enco
{

void eliminate_free_instr(coco::Module *m)
{
  for (auto ins : free_instrs(m))
  {
    destroy(ins);
  }
}

} // namespace enco
