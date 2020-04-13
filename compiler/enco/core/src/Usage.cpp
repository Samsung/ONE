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

#include "Usage.h"

namespace enco
{

std::set<coco::Block *> readers(const coco::Bag *bag)
{
  std::set<coco::Block *> res;

  for (auto read : coco::readers(bag))
  {
    assert(read != nullptr);
    auto instr = read->loc();
    assert(instr != nullptr);
    auto block = instr->parent();
    assert(block != nullptr);

    res.insert(block);
  }

  return res;
}

std::set<coco::Block *> updaters(const coco::Bag *bag)
{
  std::set<coco::Block *> res;

  for (auto update : coco::updaters(bag))
  {
    assert(update != nullptr);
    auto instr = update->loc();
    assert(instr != nullptr);
    auto block = instr->parent();
    assert(block != nullptr);

    res.insert(block);
  }

  return res;
}

} // namespace enco
