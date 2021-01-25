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

#include <luci/IR/CircleNodes.h>
#include <luci/Service/CircleShapeInference.h>

#include <loco/IR/TensorShape.h>

#include <oops/InternalExn.h>

#include <gtest/gtest.h>

TEST(CircleShapeInferenceRuleTest, CircleGatherNd_simple)
{
  luci::CircleInput input;
  luci::CircleConst indices_const;
  luci::CircleGatherNd gather_nd;

  input.shape({1, 4, 4, 3});
  indices_const.shape({1, 2, 3});

  input.shape_status(luci::ShapeStatus::VALID);
  indices_const.shape_status(luci::ShapeStatus::VALID);

  gather_nd.params(&input);
  gather_nd.indices(&indices_const);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&gather_nd, shape));
  ASSERT_EQ(3, shape.rank());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
}

TEST(CircleShapeInferenceRuleTest, CircleGatherNd_slices)
{
  luci::CircleInput input;
  luci::CircleConst indices_const;
  luci::CircleGatherNd gather_nd;

  input.shape({1, 4, 4, 3});
  indices_const.shape({1, 2, 1});

  input.shape_status(luci::ShapeStatus::VALID);
  indices_const.shape_status(luci::ShapeStatus::VALID);

  gather_nd.params(&input);
  gather_nd.indices(&indices_const);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&gather_nd, shape));
  ASSERT_EQ(5, shape.rank());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(4, shape.dim(2).value());
  ASSERT_EQ(4, shape.dim(3).value());
  ASSERT_EQ(3, shape.dim(4).value());
}

TEST(CircleShapeInferenceRuleTest, CircleGatherNd_NEG)
{
  luci::CircleInput input;
  luci::CircleConst indices_const;
  luci::CircleGatherNd gather_nd;

  input.shape({1, 4, 4, 3});
  indices_const.shape({1, 2, 5});

  input.shape_status(luci::ShapeStatus::VALID);
  indices_const.shape_status(luci::ShapeStatus::VALID);

  gather_nd.params(&input);
  gather_nd.indices(&indices_const);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_THROW(shape_inf_rule.infer(&gather_nd, shape), oops::InternalExn);
}
