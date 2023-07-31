/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <luci/Service/CircleTypeInference.h>

#include <loco/IR/TensorShape.h>

#include <gtest/gtest.h>

TEST(ShapeRuleTest, simple_hardswish)
{
  luci::CircleInput input;
  luci::CircleHardSwish hard_swish;

  input.shape({3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  hard_swish.features(&input);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&hard_swish, shape));
  ASSERT_EQ(2, shape.rank());
  ASSERT_EQ(3, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
}

TEST(DataTypeRuleTest, simple_hardswish)
{
  luci::CircleInput input;
  luci::CircleHardSwish hard_swish;

  input.dtype(loco::DataType::S32);

  hard_swish.features(&input);

  loco::DataType dtype;
  luci::tinf::Rule type_inf_rule;

  ASSERT_TRUE(type_inf_rule.infer(&hard_swish, dtype));
  ASSERT_EQ(loco::DataType::S32, dtype);
}

TEST(CloneNodeTest, clone_HardSwish)
{
  auto g = loco::make_graph();
  auto node_hardswish = g->nodes()->create<luci::CircleHardSwish>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_hardswish, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_hardswish = dynamic_cast<luci::CircleHardSwish *>(cloned);
  ASSERT_NE(nullptr, cloned_hardswish);
}
