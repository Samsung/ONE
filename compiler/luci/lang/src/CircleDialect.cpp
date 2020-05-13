/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CircleDialect.h"
#include "luci/IR/Nodes/CircleInput.h"
#include "luci/IR/Nodes/CircleOutput.h"

#include <loco/IR/Graph.h>
#include <loco/IR/GraphInputIndex.h>
#include <loco/IR/GraphOutputIndex.h>

#include <logo/DeadNodeQueryService.h>

#include <cassert>
#include <memory>

namespace
{

struct GiiQueryServiceImpl final : public loco::GraphInputIndexQueryService
{
  bool associated(const loco::Node *node) const final
  {
    if (auto circleinput = dynamic_cast<const luci::CircleInput *>(node))
    {
      return circleinput->indexed();
    }
    return false;
  }

  loco::GraphOutputIndex index(const loco::Node *node) const final
  {
    assert(associated(node));
    auto circleinput = loco::must_cast<const luci::CircleInput *>(node);
    return circleinput->index();
  }
};

struct GoiQueryServiceImpl final : public loco::GraphOutputIndexQueryService
{
  bool associated(const loco::Node *node) const final
  {
    if (auto circleoutput = dynamic_cast<const luci::CircleOutput *>(node))
    {
      return circleoutput->indexed();
    }
    return false;
  }

  loco::GraphOutputIndex index(const loco::Node *node) const final
  {
    assert(associated(node));
    auto circleoutput = loco::must_cast<const luci::CircleOutput *>(node);
    return circleoutput->index();
  }
};

struct DeadNodeQueryServiceImpl final : public logo::DeadNodeQueryService
{
  bool isDeadNode(loco::Node *node) final
  {
    auto g = node->graph();
    auto input_nodes_vec = loco::input_nodes(g);
    auto output_nodes_vec = loco::output_nodes(g);

    auto input_nodes = std::set<loco::Node *>(input_nodes_vec.begin(), input_nodes_vec.end());
    auto output_nodes = std::set<loco::Node *>(output_nodes_vec.begin(), output_nodes_vec.end());
    auto active_nodes = loco::active_nodes(output_nodes_vec);

    if (active_nodes.find(node) != active_nodes.end())
      return false;
    // input and output nodes are not dead node even if it is not active.
    if (input_nodes.find(node) != input_nodes.end())
      return false;
    if (output_nodes.find(node) != output_nodes.end())
      return false;

    return true;
  }
};

} // namespace

namespace luci
{

CircleDialect::CircleDialect()
{
  service<loco::GraphInputIndexQueryService>(std::make_unique<GiiQueryServiceImpl>());
  service<loco::GraphOutputIndexQueryService>(std::make_unique<GoiQueryServiceImpl>());
  service<logo::DeadNodeQueryService>(std::make_unique<DeadNodeQueryServiceImpl>());
}

loco::Dialect *CircleDialect::get(void)
{
  static CircleDialect d;
  return &d;
}

} // namespace luci
