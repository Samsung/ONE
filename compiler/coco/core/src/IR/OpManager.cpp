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

#include "coco/IR/OpManager.h"

#include <memory>
#include <cassert>
#include <queue>
#include <set>

using std::make_unique;

namespace coco
{

OpManager::~OpManager()
{
  std::set<coco::Op *> roots;

  for (uint32_t n = 0; n < size(); ++n)
  {
    auto op = at(n);

    if (op->up() != nullptr)
    {
      continue;
    }

    roots.insert(op);
  }

  for (const auto &op : roots)
  {
    destroy_all(op);
  }
}

//
// Each Op class SHOULD be default constructible
//
#define OP(Name)                                  \
  template <> Name *OpManager::create<Name>(void) \
  {                                               \
    auto op = make_unique<Name>();                \
    modulize(op.get());                           \
    return take(std::move(op));                   \
  }
#include "coco/IR/Op.lst"
#undef OP

void OpManager::destroy(Op *op)
{
  assert(op->parent() == nullptr);
  release(op);
}

void OpManager::destroy_all(Op *op)
{
  assert(op->parent() == nullptr);
  assert(op->up() == nullptr);

  std::queue<coco::Op *> q;

  q.emplace(op);

  while (q.size() > 0)
  {
    auto cur = q.front();
    q.pop();

    // Insert child op nodes
    for (uint32_t n = 0; n < cur->arity(); ++n)
    {
      if (auto child = cur->arg(n))
      {
        q.emplace(child);
      }
    }

    // Destroy the current op node
    destroy(cur);
  }
}

} // namespace coco
