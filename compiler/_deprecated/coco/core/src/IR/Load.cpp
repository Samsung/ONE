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

#include "coco/IR/Ops.h"

#include <cassert>

namespace coco
{

Load::Load() : _obj{this}
{
  // DO NOTHING
}

uint32_t Load::arity(void) const
{
  // Load has no child Op
  return 0;
}

Op *Load::arg(uint32_t) const
{
  assert(!"Load has no argument");
  return nullptr;
}

std::set<Object *> Load::uses(void) const
{
  std::set<Object *> res;

  if (auto obj = object())
  {
    res.insert(obj);
  }

  return res;
}

} // namespace coco
