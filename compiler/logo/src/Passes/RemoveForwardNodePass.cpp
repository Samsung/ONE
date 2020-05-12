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

#include <logo/RemoveForwardNodePass.h>

#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>

#include <set>

namespace logo
{

bool RemoveForwardNodePass::run(loco::Graph *g)
{
  struct Collector final : public loco::CanonicalNodeMutableVisitor<void>
  {
    void visit(loco::Forward *node) final
    {
      if (node->input() != nullptr)
      {
        candidates.insert(node);
      }
    }

    void visit(loco::Node *) final { return; }

    std::set<loco::Forward *> candidates;
  };

  Collector collector;

  for (auto node : loco::all_nodes(g))
  {
    if (node->dialect() == loco::CanonicalDialect::get())
    {
      auto canonical_node = loco::must_cast<loco::CanonicalNode *>(node);
      canonical_node->accept(&collector);
    }
  }

  for (auto node : collector.candidates)
  {
    replace(node).with(node->input());
    node->input(nullptr);
  }

  return collector.candidates.size() > 0;
}

} // namespace logo
