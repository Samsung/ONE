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

#include "luci/Service/CircleNodeClone.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/CircleShapeInference.h>

#include <loco/IR/TensorShape.h>

#include <gtest/gtest.h>

TEST(ShapeRuleTest, squeeze_simple)
{
  luci::CircleInput input;
  luci::CircleSqueeze squeeze;

  input.shape({1, 4, 3, 1});
  input.shape_status(luci::ShapeStatus::VALID);

  squeeze.input(&input);
  squeeze.squeeze_dims({0});

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&squeeze, shape));
  ASSERT_EQ(3, shape.rank());
  ASSERT_EQ(4, shape.dim(0).value());
  ASSERT_EQ(3, shape.dim(1).value());
  ASSERT_EQ(1, shape.dim(2).value());
}

TEST(ShapeRuleTest, squeeze_all)
{
  luci::CircleInput input;
  luci::CircleSqueeze squeeze;

  input.shape({1, 4, 3, 1});
  input.shape_status(luci::ShapeStatus::VALID);

  squeeze.input(&input);
  squeeze.squeeze_dims({});

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&squeeze, shape));
  ASSERT_EQ(2, shape.rank());
  ASSERT_EQ(4, shape.dim(0).value());
  ASSERT_EQ(3, shape.dim(1).value());
}

TEST(CloneNodeTest, clone_Squeeze)
{
  auto g = loco::make_graph();
  auto node_squ = g->nodes()->create<luci::CircleSqueeze>();
  node_squ->squeeze_dims({2, 3});

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_squ, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_squ = dynamic_cast<luci::CircleSqueeze *>(cloned);
  ASSERT_NE(nullptr, cloned_squ);
  ASSERT_EQ(node_squ->squeeze_dims().size(), cloned_squ->squeeze_dims().size());
  for (size_t s = 0; s < node_squ->squeeze_dims().size(); ++s)
    ASSERT_EQ(node_squ->squeeze_dims().at(s), cloned_squ->squeeze_dims().at(s));
}
