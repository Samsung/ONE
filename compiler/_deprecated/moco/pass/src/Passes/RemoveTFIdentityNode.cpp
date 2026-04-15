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

#include "moco/Pass/Passes/RemoveTFIdentityNode.h"

#include <moco/IR/TFDialect.h>
#include <moco/IR/TFNode.h>

#include <set>

namespace moco
{

bool RemoveTFIdentityNode::run(loco::Graph *g)
{
  struct Collector final : public moco::TFNodeMutableVisitor<void>
  {
    void visit(moco::TFIdentity *node) final
    {
      if (node->input() != nullptr)
      {
        candidates.insert(node);
      }
    }

    void visit(moco::TFNode *) final { return; }

    std::set<moco::TFIdentity *> candidates;
  };

  Collector collector;

  for (auto node : loco::all_nodes(g))
  {
    if (node->dialect() == moco::TFDialect::get())
    {
      auto tf_node = dynamic_cast<moco::TFNode *>(node);
      // NOTE our analysis tool reports an error for tf_node may be nullptr
      if (tf_node != nullptr)
        tf_node->accept(&collector);
    }
  }

  for (auto node : collector.candidates)
  {
    replace(node).with(node->input());
    node->input(nullptr);
  }

  return collector.candidates.size() > 0;
}

} // namespace moco
