/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loco/IR/Node.h"
#include "loco/IR/Use.h"

#include <cassert>

namespace loco
{

Node::~Node()
{
  // To detect dangling references
  assert(_uses.size() == 0);
}

std::set<Node *> preds(const Node *node)
{
  std::set<Node *> res;

  for (uint32_t n = 0; n < node->arity(); ++n)
  {
    if (auto pred = node->arg(n))
    {
      res.insert(pred);
    }
  }

  return res;
}

std::set<Node *> succs(const Node *node)
{
  std::set<Node *> res;

  for (auto use : node->_uses)
  {
    auto user = use->user();
    assert(user != nullptr);
    res.insert(user);
  }

  return res;
}

Subst<SubstQualifier::Default>::Subst(Node *from) : _from{from}
{
  // _from SHOULD be valid
  assert(_from != nullptr);
}

void Subst<SubstQualifier::Default>::with(Node *into) const
{
  if (_from == into)
  {
    return;
  }

  auto *uses = &(_from->_uses);

  while (!uses->empty())
  {
    auto use = *(uses->begin());
    use->node(into);
  }
}

Subst<SubstQualifier::Default> replace(Node *node)
{
  // Let's create Subst<SubstQualifier::Default>!
  return Subst<SubstQualifier::Default>{node};
}

} // namespace loco
