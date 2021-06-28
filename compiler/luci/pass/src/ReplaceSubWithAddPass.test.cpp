/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ReplaceSubWithAddPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Simple graph for test
 *
 *  BEFORE
 * 
 *   [lhs] ------------+
 *                     +-- [Sub] --
 *   [rhs_const] ------+
 *
 *  AFTER
 * 
 *   [lhs] ------------+
 *                     +-- [Add] --
 *   [neg_rhs_const] --+
 */
class SimpleGraph
{
public:
  SimpleGraph()
  {
    lhs = g.nodes()->create<luci::CircleInput>();
    rhs_const = g.nodes()->create<luci::CircleConst>();
    sub = g.nodes()->create<luci::CircleSub>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    lhs->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    lhs->dtype(loco::DataType::FLOAT32);
    rhs_const->dtype(loco::DataType::FLOAT32);
    sub->dtype(loco::DataType::FLOAT32);
    output->dtype(loco::DataType::FLOAT32);

    lhs->shape({1, 3, 4, 5});
    rhs_const->shape({}); // scalar
    sub->shape({1, 3, 4, 5});
    output->shape({1, 3, 4, 5});

    rhs_const->size<loco::DataType::FLOAT32>(1);
    rhs_const->at<loco::DataType::FLOAT32>(0) = 1.1;

    sub->x(lhs);
    sub->y(rhs_const);
    output->from(sub);

    lhs->name("lhs");
    rhs_const->name("rhs_const");
    sub->name("sub");
    output->name("output");
  }

public:
  loco::Graph g;
  luci::CircleInput *lhs = nullptr;
  luci::CircleConst *rhs_const = nullptr;
  luci::CircleSub *sub = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST(ReplaceSubWithAdd, name)
{
  luci::ReplaceSubWithAddPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(ReplaceSubWithAdd, simple)
{
  SimpleGraph g;

  luci::ReplaceSubWithAddPass pass;
  while (pass.run(&g.g))
    ;

  auto add = dynamic_cast<luci::CircleAdd *>(g.output->from());
  EXPECT_NE(nullptr, add);

  auto neg_rhs_const = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, neg_rhs_const);
  EXPECT_EQ(0, neg_rhs_const->rank());
  EXPECT_FLOAT_EQ(-1.1, neg_rhs_const->at<loco::DataType::FLOAT32>(0));
}

TEST(ReplaceSubWithAdd, wrong_op_NEG)
{
  SimpleGraph g;

  auto mul = g.g.nodes()->create<luci::CircleMul>();
  mul->x(g.sub->x());
  mul->y(g.sub->y());
  loco::replace(g.sub).with(mul);

  luci::ReplaceSubWithAddPass pass;
  auto changed = pass.run(&g.g);

  EXPECT_EQ(false, changed);
}
