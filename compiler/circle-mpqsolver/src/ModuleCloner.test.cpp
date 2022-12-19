/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <gtest/gtest.h>

#include "TestHelper.h"
#include "ModuleCloner.h"

#include <luci/IR/CircleNodeDecl.h>

namespace
{

class AddGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    _add = _g->nodes()->create<luci::CircleAdd>();
    auto beta = _g->nodes()->create<luci::CircleConst>();

    _add->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);

    _add->shape({1, _channel_size, _width, _height});
    beta->shape({1, _channel_size, _width, _height});

    _add->fusedActivationFunction(luci::FusedActFunc::NONE);

    _add->x(input);
    _add->y(beta);

    _add->name("add");
    beta->name("beta");

    return _add;
  }

public:
  luci::CircleAdd *_add = nullptr;
};

} // namespace

TEST(CircleMPQSolverModuleClonerTest, verifyResultsTest)
{
  auto m = luci::make_module();
  AddGraph g;
  g.init();
  auto all_nodes = loco::all_nodes(g._g.get());
  auto add = g._add;
  g.transfer_to(m.get());

  auto cloned = mpqsolver::ModuleCloner::clone(m.get());
  EXPECT_TRUE(cloned->size() == 1);
  auto cloned_graph = cloned->graph(0);
  auto all_cloned_nodes = loco::all_nodes(cloned_graph);
  EXPECT_TRUE(all_nodes.size() == all_cloned_nodes.size());
  bool found = false;
  for (auto node : all_cloned_nodes)
  {
    auto ci_node = dynamic_cast<luci::CircleNode *>(node);
    EXPECT_TRUE(ci_node != nullptr);
    if (ci_node->name() == add->name())
    {
      auto cloned_add = dynamic_cast<luci::CircleAdd *>(ci_node);
      EXPECT_TRUE(cloned_add != nullptr);
      EXPECT_TRUE(add->fusedActivationFunction() == cloned_add->fusedActivationFunction());
      EXPECT_TRUE(cloned_add->rank() == add->rank());
      for (uint32_t dim = 0; dim < add->rank(); ++dim)
      {
        auto cloned_dimension = cloned_add->dim(dim);
        auto dimension = add->dim(dim);
        EXPECT_TRUE(cloned_dimension.known() == dimension.known());
        EXPECT_TRUE(cloned_dimension.value() == dimension.value());
      }
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(CircleMPQSolverModuleClonerTest, verifyResultsTest_NEG)
{
  auto m = luci::make_module();
  auto cloned = mpqsolver::ModuleCloner::clone(m.get());
  EXPECT_TRUE(cloned->size() == 0);
}
