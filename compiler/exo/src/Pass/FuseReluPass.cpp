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

#include "FuseReluPass.h"

#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLDialect.h"
#include "Dialect/IR/TFLNodeVisitor.h"

#include <set>

namespace
{

bool is_pred_fusable(loco::Node *node)
{
  using namespace locoex;

  auto fusable_node = dynamic_cast<TFLNodeMixin<TFLNodeTrait::FusedActFunc> *>(node);

  return (fusable_node and fusable_node->fusedActivationFunction() == FusedActFunc::NONE);
};

struct Collector final : public locoex::TFLNodeMutableVisitor<void>
{
  void visit(locoex::TFLRelu *node) final
  {
    if (is_pred_fusable(node->features()))
      candidates.insert(node);
  }

  void visit(locoex::TFLRelu6 *node) final
  {
    if (is_pred_fusable(node->features()))
      candidates.insert(node);
  }

  void visit(locoex::TFLNode *) final { return; }

  std::set<locoex::TFLNode *> candidates;
};

void set_activation_fusion(loco::Node *node, locoex::FusedActFunc f)
{
  using namespace locoex;

  if (auto fusable_node = dynamic_cast<TFLNodeMixin<TFLNodeTrait::FusedActFunc> *>(node))
    fusable_node->fusedActivationFunction(f);
  else
    assert(false);
}

struct Performer final : public locoex::TFLNodeMutableVisitor<void>
{
  void visit(locoex::TFLRelu *the_relu) final
  {
    set_activation_fusion(the_relu->features(), locoex::FusedActFunc::RELU);

    loco::replace(the_relu).with(the_relu->features());
    the_relu->features(nullptr);
  }

  void visit(locoex::TFLRelu6 *the_relu6) final
  {
    set_activation_fusion(the_relu6->features(), locoex::FusedActFunc::RELU6);

    loco::replace(the_relu6).with(the_relu6->features());
    the_relu6->features(nullptr);
  }

  void visit(locoex::TFLNode *) final { assert(false && "should not be called"); }
};

} // namespace

namespace exo
{

bool FuseReluPass::run(loco::Graph *g)
{
  Collector collector;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (node->dialect() == locoex::TFLDialect::get())
    {
      auto tfl_node = loco::must_cast<locoex::TFLNode *>(node);
      tfl_node->accept(&collector);
    }
  }

  Performer performer;

  for (auto node : collector.candidates)
  {
    node->accept(&performer);
  }

  return collector.candidates.size() > 0;
}

} // namespace exo
