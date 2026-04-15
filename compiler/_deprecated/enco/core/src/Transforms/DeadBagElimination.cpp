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

#include "DeadBagElimination.h"

#include <set>

namespace
{

/// @brief Return true if a given bag is marked as either input or output
bool is_public(const coco::Bag *b) { return b->isInput() || b->isOutput(); }

/// @brief Return the set of "dead" bags in a given module
std::set<coco::Bag *> dead_bags(const coco::Module *m)
{
  std::set<coco::Bag *> res;

  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);

    if (coco::readers(bag).empty() && !is_public(bag))
    {
      res.insert(bag);
    }
  }

  return res;
}

} // namespace

namespace enco
{

void eliminate_dead_bag(enco::Code *code)
{
  auto m = code->module();

  // Destroy a dead bag and its updaters
  for (auto bag : dead_bags(m))
  {
    for (auto updater : coco::updaters(bag))
    {
      auto ins = updater->loc();

      assert(ins != nullptr);

      ins->detach();
      m->entity()->instr()->destroy(ins);
    }

    bag->replaceWith(nullptr);
    m->entity()->bag()->destroy(bag);
  }
}

} // namespace enco
