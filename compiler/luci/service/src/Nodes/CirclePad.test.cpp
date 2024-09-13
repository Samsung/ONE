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

#include "luci/Service/CircleShapeInference.h"

#include <gtest/gtest.h>

TEST(CloneNodeTest, clone_Pad)
{
  auto g = loco::make_graph();
  auto node_pad = g->nodes()->create<luci::CirclePad>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_pad, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_pad = dynamic_cast<luci::CirclePad *>(cloned);
  ASSERT_NE(nullptr, cloned_pad);
}

TEST(ShapeRuleTest, pad_dynamic_shape)
{
  auto g = loco::make_graph();
  auto node_pad = g->nodes()->create<luci::CirclePad>();

  auto node_paddings = g->nodes()->create<luci::CircleConst>();
  auto node_input = g->nodes()->create<luci::CircleInput>();

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  node_input->shape({1, 2, 3, 4});
  node_input->shape_status(luci::ShapeStatus::VALID);
  node_input->dim(2).unset();

  node_paddings->dtype(loco::DataType::S64);
  node_paddings->shape({4, 2});
  node_paddings->shape_status(luci::ShapeStatus::VALID);

  const loco::DataType S64 = loco::DataType::S64;
  uint32_t t = 64 * 8;
  node_paddings->size<S64>(t);

  node_pad->input(node_input);
  node_pad->paddings(node_paddings);

  ASSERT_TRUE(shape_inf_rule.infer(node_pad, shape));
  ASSERT_EQ(shape.rank(), 4);
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_FALSE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());

  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(0, shape.dim(2).value());
  ASSERT_EQ(4, shape.dim(3).value());
}

TEST(ShapeRuleTest, pad_without_padding_NEG)
{
  auto g = loco::make_graph();
  auto node_pad = g->nodes()->create<luci::CirclePad>();
  auto node_input = g->nodes()->create<luci::CircleInput>();

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  node_input->shape({1, 2, 3, 4});
  node_input->shape_status(luci::ShapeStatus::VALID);
  node_input->dim(2).unset();

  node_pad->input(node_input);

  ASSERT_ANY_THROW(shape_inf_rule.infer(node_pad, shape));
}

TEST(ShapeRuleTest, pad_non_const_paddings)
{
  auto g = loco::make_graph();
  auto node_pad = g->nodes()->create<luci::CirclePad>();

  auto node_paddings = g->nodes()->create<luci::CircleInput>();
  auto node_input = g->nodes()->create<luci::CircleInput>();

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  node_input->shape({1, 2, 3, 4});
  node_input->shape_status(luci::ShapeStatus::VALID);

  node_paddings->dtype(loco::DataType::S64);
  node_paddings->shape({4, 2});
  node_paddings->shape_status(luci::ShapeStatus::VALID);

  node_pad->input(node_input);
  node_pad->paddings(node_paddings);

  ASSERT_TRUE(shape_inf_rule.infer(node_pad, shape));
  ASSERT_EQ(shape.rank(), 4);
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_FALSE(shape.dim(1).known());
  ASSERT_FALSE(shape.dim(2).known());
  ASSERT_FALSE(shape.dim(3).known());

  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(0, shape.dim(2).value());
  ASSERT_EQ(0, shape.dim(3).value());
}

TEST(ShapeRuleTest, paddings_invalid_rank_NEG)
{
  auto g = loco::make_graph();
  auto node_pad = g->nodes()->create<luci::CirclePad>();

  auto node_paddings = g->nodes()->create<luci::CircleConst>();
  auto node_input = g->nodes()->create<luci::CircleInput>();

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  node_input->shape({1, 2, 3, 4});
  node_input->shape_status(luci::ShapeStatus::VALID);

  node_paddings->dtype(loco::DataType::S64);
  node_paddings->shape({4, 2, 3});
  node_paddings->shape_status(luci::ShapeStatus::VALID);

  const loco::DataType S64 = loco::DataType::S64;
  uint32_t t = 64 * 8;
  node_paddings->size<S64>(t);

  node_pad->input(node_input);
  node_pad->paddings(node_paddings);

  ASSERT_ANY_THROW(shape_inf_rule.infer(node_pad, shape));
}

TEST(ShapeRuleTest, paddings_invalid_shape_1_NEG)
{
  auto g = loco::make_graph();
  auto node_pad = g->nodes()->create<luci::CirclePad>();

  auto node_paddings = g->nodes()->create<luci::CircleConst>();
  auto node_input = g->nodes()->create<luci::CircleInput>();

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  node_input->shape({1, 2, 3, 4});
  node_input->shape_status(luci::ShapeStatus::VALID);

  node_paddings->dtype(loco::DataType::S64);
  node_paddings->shape({4, 4});
  node_paddings->shape_status(luci::ShapeStatus::VALID);

  const loco::DataType S64 = loco::DataType::S64;
  uint32_t t = 64 * 8;
  node_paddings->size<S64>(t);

  node_pad->input(node_input);
  node_pad->paddings(node_paddings);

  ASSERT_ANY_THROW(shape_inf_rule.infer(node_pad, shape));
}

TEST(ShapeRuleTest, paddings_invalid_shape_2_NEG)
{
  auto g = loco::make_graph();
  auto node_pad = g->nodes()->create<luci::CirclePad>();

  auto node_paddings = g->nodes()->create<luci::CircleConst>();
  auto node_input = g->nodes()->create<luci::CircleInput>();

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  node_input->shape({1, 2, 3, 4});
  node_input->shape_status(luci::ShapeStatus::VALID);

  node_paddings->dtype(loco::DataType::S64);
  node_paddings->shape({5, 2});
  node_paddings->shape_status(luci::ShapeStatus::VALID);

  const loco::DataType S64 = loco::DataType::S64;
  uint32_t t = 64 * 8;
  node_paddings->size<S64>(t);

  node_pad->input(node_input);
  node_pad->paddings(node_paddings);

  ASSERT_ANY_THROW(shape_inf_rule.infer(node_pad, shape));
}
