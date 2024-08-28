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

TEST(CloneNodeTest, clone_BatchMatMul)
{
  auto g = loco::make_graph();
  auto node_bmm = g->nodes()->create<luci::CircleBatchMatMul>();
  node_bmm->adj_x(true);
  node_bmm->adj_y(true);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_bmm, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_bmm = dynamic_cast<luci::CircleBatchMatMul *>(cloned);
  ASSERT_NE(nullptr, cloned_bmm);
  ASSERT_EQ(node_bmm->adj_x(), cloned_bmm->adj_x());
  ASSERT_EQ(node_bmm->adj_y(), cloned_bmm->adj_y());
}

TEST(ShapeRuleTest, bmm_broadcast_known_dim_1)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({2, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({1, 3, 5});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&bmm, shape));

  // (2, 4, 3) x (1, 3, 5) -> (2, 4, 5)
  // output shape should be (2, 4, 5)
  ASSERT_EQ(3, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_EQ(2, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(5, shape.dim(2).value());
}

TEST(ShapeRuleTest, bmm_broadcast_known_dim_2)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({5, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({3, 8});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&bmm, shape));

  // (5, 4, 3) x (3, 8) -> (5, 3, 8)
  // output shape should be (5, 4, 8)
  ASSERT_EQ(3, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_EQ(5, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(8, shape.dim(2).value());
}

TEST(ShapeRuleTest, bmm_with_dynamic_shape_1)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({1, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);
  input_x.dim(0).unset(); // {0, 4, 3}

  input_y.shape({2, 5, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&bmm, shape));

  // (0, 4, 3) x (2, 5, 3, 7) -> (2, 5, 4, 7)
  // output shape should be (2, 5, 4, 7)
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(2, shape.dim(0).value());
  ASSERT_EQ(5, shape.dim(1).value());
  ASSERT_EQ(4, shape.dim(2).value());
  ASSERT_EQ(7, shape.dim(3).value());
}

TEST(ShapeRuleTest, bmm_with_dynamic_shape_2)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({1, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);
  input_x.dim(0).unset(); // {0, 4, 3}

  input_y.shape({2, 5, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);
  input_y.dim(0).unset();
  input_y.dim(1).unset(); // {0, 0, 3, 7}

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&bmm, shape));

  // (0, 4, 3) x (0, 0, 3, 7) -> (0, 0, 4, 7)
  // output shape should be (0, 0, 4, 7)
  ASSERT_EQ(4, shape.rank());
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_FALSE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(4, shape.dim(2).value());
  ASSERT_EQ(7, shape.dim(3).value());
}

TEST(ShapeRuleTest, bmm_with_dynamic_shape_3)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({1, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({1, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);
  input_y.dim(1).unset(); // {1, 0, 7}

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&bmm, shape));

  // (1, 4, 3) x (1, 0, 7) -> (1, 4, 7)
  // output shape should be (1, 4, 7)
  ASSERT_EQ(3, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(7, shape.dim(2).value());
}

TEST(ShapeRuleTest, bmm_with_dynamic_shape_4)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({2, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({2, 2, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);
  input_y.dim(1).unset(); // {2, 0, 3, 7}

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&bmm, shape));

  // (2, 4, 3) x (2, 0, 3, 7) -> (2, 2, 4, 7)
  // output shape should be (2, 2, 4, 7)
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(2, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(4, shape.dim(2).value());
  ASSERT_EQ(7, shape.dim(3).value());
}

TEST(ShapeRuleTest, bmm_not_broadcastable_1_NEG)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({2, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({3, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  // (2, 4, 3) x (3, 3, 7)
  //  ^           ^
  // => error, batch dimension failed to broadcast
  ASSERT_ANY_THROW(shape_inf_rule.infer(&bmm, shape));
}

TEST(ShapeRuleTest, bmm_not_broadcastable_2_NEG)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({2, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({1, 3, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  // (2, 4, 3) x (1, 3, 3, 7)
  //  ^           ^  ^
  // => error, batch dimension failed to broadcast
  ASSERT_ANY_THROW(shape_inf_rule.infer(&bmm, shape));
}

TEST(ShapeRuleTest, bmm_not_broadcastable_3_NEG)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({2, 4, 3});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({4, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  // (2, 4, 3) x (4, 3, 7)
  //  ^           ^
  // => error, batch dimension failed to broadcast
  ASSERT_ANY_THROW(shape_inf_rule.infer(&bmm, shape));
}

TEST(ShapeRuleTest, bmm_mismatch_dim_1_NEG)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({1, 4, 4});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({1, 3, 7});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  // (1, 4, 4) x (1, 3, 7)
  //        ^        ^
  // => error, matmul hidden dim should be same
  ASSERT_ANY_THROW(shape_inf_rule.infer(&bmm, shape));
}

TEST(ShapeRuleTest, bmm_mismatch_dim_2_NEG)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({2, 40, 40});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({1, 30, 70});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  // (2, 40, 40) x (1, 30, 70)
  //         ^         ^
  // => error, matmul hidden dim should be same
  ASSERT_ANY_THROW(shape_inf_rule.infer(&bmm, shape));
}

TEST(ShapeRuleTest, bmm_1D_rank_NEG)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({2});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({2, 4});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  // (2) x (2, 4)
  //  ^
  // => error, x_rank should be >= 2
  ASSERT_ANY_THROW(shape_inf_rule.infer(&bmm, shape));
}

TEST(ShapeRuleTest, bmm_empty_input_NEG)
{
  luci::CircleInput input_x;
  luci::CircleInput input_y;
  luci::CircleBatchMatMul bmm;

  input_x.shape({0, 2});
  input_x.shape_status(luci::ShapeStatus::VALID);

  input_y.shape({2, 4});
  input_y.shape_status(luci::ShapeStatus::VALID);

  bmm.x(&input_x);
  bmm.y(&input_y);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  // (0, 2) x (2, 4)
  //  ^
  // => error, x should not be empty
  ASSERT_ANY_THROW(shape_inf_rule.infer(&bmm, shape));
}
