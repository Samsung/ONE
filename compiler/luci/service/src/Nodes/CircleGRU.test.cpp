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

TEST(ShapeRuleTest, simple_circle_gru)
{
  luci::CircleInput input;
  luci::CircleConst hidden_hidden;
  luci::CircleConst hidden_input;
  luci::CircleConst state;
  luci::CircleGRU circle_gru;

  input.shape({10, 1, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  hidden_hidden.shape({7, 32});
  hidden_hidden.shape_status(luci::ShapeStatus::VALID);

  hidden_input.shape({7, 4});
  hidden_input.shape_status(luci::ShapeStatus::VALID);

  state.shape({1, 32});
  state.shape_status(luci::ShapeStatus::VALID);

  circle_gru.input(&input);
  circle_gru.hidden_hidden(&hidden_hidden);
  circle_gru.hidden_input(&hidden_input);
  circle_gru.state(&state);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&circle_gru, shape));
  ASSERT_EQ(3, shape.rank());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(1, shape.dim(1).value());
  ASSERT_EQ(32, shape.dim(2).value());
}

TEST(DataTypeRuleTest, simple_circle_gru)
{
  luci::CircleInput input;
  luci::CircleConst hidden_hidden;
  luci::CircleConst hidden_input;
  luci::CircleConst state;
  luci::CircleGRU circle_gru;

  input.dtype(loco::DataType::FLOAT32);
  hidden_hidden.dtype(loco::DataType::FLOAT32);
  hidden_input.dtype(loco::DataType::FLOAT32);
  state.dtype(loco::DataType::FLOAT32);

  circle_gru.input(&input);
  circle_gru.hidden_hidden(&hidden_hidden);
  circle_gru.hidden_input(&hidden_input);
  circle_gru.state(&state);

  loco::DataType dtype;
  luci::tinf::Rule type_inf_rule;

  ASSERT_TRUE(type_inf_rule.infer(&circle_gru, dtype));
  ASSERT_EQ(loco::DataType::FLOAT32, dtype);
}

TEST(CloneNodeTest, clone_circel_gru)
{
  auto g = loco::make_graph();
  auto node_circle_gru = g->nodes()->create<luci::CircleGRU>();
  node_circle_gru->fusedActivationFunction(luci::FusedActFunc::TANH);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_circle_gru, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_circle_gru = dynamic_cast<luci::CircleGRU *>(cloned);
  ASSERT_NE(nullptr, cloned_circle_gru);
}
