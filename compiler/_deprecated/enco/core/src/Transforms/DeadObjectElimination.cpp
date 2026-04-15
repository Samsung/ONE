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

#include "DeadObjectElimination.h"

#include <set>

namespace
{

std::set<coco::Object *> dead_objects(const coco::Module *m)
{
  std::set<coco::Object *> res;

  for (uint32_t n = 0; n < m->entity()->object()->size(); ++n)
  {
    auto obj = m->entity()->object()->at(n);

    if (auto bag = obj->bag())
    {
      if (coco::readers(bag).empty() && !(bag->isOutput()))
      {
        res.insert(obj);
      }
    }
    else
    {
      // NOTE Just in case if there are Objects not related to Bags
      if (obj->uses()->size() == 0)
      {
        res.insert(obj);
      }
    }
  }

  return res;
}

} // namespace

namespace enco
{

void eliminate_dead_object(enco::Code *code)
{
  auto m = code->module();

  // Destroy a dead object and its producer
  for (auto obj : dead_objects(m))
  {
    if (auto producer = coco::producer(obj))
    {
      auto ins = producer->loc();
      assert(ins != nullptr);

      ins->detach();
      m->entity()->instr()->destroy(ins);
    }

    m->entity()->object()->destroy(obj);
  }
}

} // namespace enco
