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

#include "IRUtils.h"

#include <cassert>

namespace enco
{

/**
 * @brief Substitute all the USE occurrences of an object with another object
 * @param from  Object to be replaced
 * @param into  Object to be used instead
 * NOTE This maybe used when something like -- 'from' will be removed so we need
 *      to replace object Consumers that use 'from' to 'into'
 * EXAMPLE
 *      {
 *        subst(child, bigone);
 *        m->entity()->object()->destroy(child);
 *      }
 *      This code will change all the Consumers that use 'child' to 'bigone' and
 *      destroy the 'child' object.
 */
void subst(coco::Object *from, coco::Object *into)
{
  assert(from != into);

  while (!from->uses()->empty())
  {
    auto use = *(from->uses()->begin());

    use->value(into);
  }
}

std::vector<coco::Instr *> instr_sequence(coco::Module *m)
{
  std::vector<coco::Instr *> res;

  for (auto B = m->block()->head(); B; B = B->next())
  {
    for (auto I = B->instr()->head(); I; I = I->next())
    {
      res.emplace_back(I);
    }
  }

  return res;
}

} // namespace enco
