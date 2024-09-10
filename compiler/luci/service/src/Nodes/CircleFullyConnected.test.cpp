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

TEST(CloneNodeTest, clone_FullyConnected)
{
  auto g = loco::make_graph();
  auto node_fc = g->nodes()->create<luci::CircleFullyConnected>();
  node_fc->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_fc->weights_format(luci::CircleFullyConnected::WeightsFormat::DEFAULT);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_fc, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_fc = dynamic_cast<luci::CircleFullyConnected *>(cloned);
  ASSERT_NE(nullptr, cloned_fc);
  ASSERT_EQ(node_fc->fusedActivationFunction(), cloned_fc->fusedActivationFunction());
  ASSERT_EQ(node_fc->weights_format(), cloned_fc->weights_format());
}

TEST(CloneNodeTest, clone_FullyConnected_fusedact_NEG)
{
  auto g = loco::make_graph();
  auto node_fc = g->nodes()->create<luci::CircleFullyConnected>();
  node_fc->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node_fc->weights_format(luci::CircleFullyConnected::WeightsFormat::DEFAULT);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_fc, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(CloneNodeTest, clone_FullyConnected_wf_NEG)
{
  auto g = loco::make_graph();
  auto node_fc = g->nodes()->create<luci::CircleFullyConnected>();
  node_fc->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_fc->weights_format(luci::CircleFullyConnected::WeightsFormat::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_fc, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(ShapeRuleTest, fully_connected_dynamic_shape_keep_dims)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 10, 15, 20});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(1).unset();

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(true);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&fully_connected, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_FALSE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());

  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(15, shape.dim(2).value());
  ASSERT_EQ(30, shape.dim(3).value());
}

TEST(ShapeRuleTest, fully_connected_last_dim_dynamic_keep_dims)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 10, 15, 20});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(3).unset();

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(true);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&fully_connected, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());

  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(10, shape.dim(1).value());
  ASSERT_EQ(15, shape.dim(2).value());
  ASSERT_EQ(30, shape.dim(3).value());
}

TEST(ShapeRuleTest, fully_connected_dynamic_shape_no_keep_dims)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 10, 15, 20});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(2).unset();

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(false);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&fully_connected, shape));
  ASSERT_EQ(2, shape.rank());
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());

  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(30, shape.dim(1).value());
}

TEST(ShapeRuleTest, fully_connected_last_dim_dynamic_no_keep_dims)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 10, 15, 20});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(3).unset();

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(false);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&fully_connected, shape));
  ASSERT_EQ(2, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());

  ASSERT_EQ(150, shape.dim(0).value());
  ASSERT_EQ(30, shape.dim(1).value());
}

TEST(ShapeRuleTest, fully_connected_all_dim_dynamic_no_keep_dims)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 10, 15, 20});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(0).unset();
  input.dim(1).unset();
  input.dim(2).unset();
  input.dim(3).unset();

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(false);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&fully_connected, shape));
  ASSERT_EQ(2, shape.rank());
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());

  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(30, shape.dim(1).value());
}

TEST(ShapeRuleTest, fully_connected_nullptr_weights_NEG)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 10, 20});
  input.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(nullptr);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(true);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&fully_connected, shape));
}

TEST(ShapeRuleTest, fully_connected_nullptr_input_NEG)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(nullptr);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(true);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&fully_connected, shape));
}

TEST(ShapeRuleTest, fully_connected_nullptr_bias_NEG)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 15, 20});
  input.shape_status(luci::ShapeStatus::VALID);

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(nullptr);
  fully_connected.keep_num_dims(true);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&fully_connected, shape));
}

TEST(ShapeRuleTest, fully_connected_undefined_bias_NEG)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape({1, 15, 20});
  input.shape_status(luci::ShapeStatus::VALID);

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::UNDEFINED);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(true);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_FALSE(shape_inf_rule.infer(&fully_connected, shape));
}

TEST(ShapeRuleTest, fully_connected_undefined_input_NEG)
{
  luci::CircleInput input;
  luci::CircleConst weights;
  luci::CircleConst bias;
  luci::CircleFullyConnected fully_connected;

  input.shape_status(luci::ShapeStatus::UNDEFINED);

  weights.shape({30, 20});
  weights.shape_status(luci::ShapeStatus::VALID);

  bias.shape_status(luci::ShapeStatus::VALID);

  fully_connected.input(&input);
  fully_connected.weights(&weights);
  fully_connected.bias(&bias);
  fully_connected.keep_num_dims(true);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_FALSE(shape_inf_rule.infer(&fully_connected, shape));
}
