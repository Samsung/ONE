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

TEST(CloneNodeTest, clone_Reshape)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  node_reshape->newShape()->rank(2);
  node_reshape->newShape()->dim(0) = 3;
  node_reshape->newShape()->dim(1) = 4;

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_reshape, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_reshape = dynamic_cast<luci::CircleReshape *>(cloned);
  ASSERT_NE(nullptr, cloned_reshape);
  ASSERT_EQ(node_reshape->newShape()->rank(), cloned_reshape->newShape()->rank());
  ASSERT_EQ(node_reshape->newShape()->dim(0), cloned_reshape->newShape()->dim(0));
  ASSERT_EQ(node_reshape->newShape()->dim(1), cloned_reshape->newShape()->dim(1));
}

TEST(ShapeRuleTest, reshape_known_dim)
{
  luci::CircleInput input;
  input.rank(3);
  input.dim(0) = 2;
  input.dim(1) = 3;
  input.dim(2) = 4;
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.rank(2);
  new_shape.dim(0) = 6;
  new_shape.dim(1) = 4;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  std::cout << "new_shape: " << &new_shape << std::endl;
  std::cout << "new_shape.rank(): " << new_shape.rank() << std::endl;
  std::cout << "new_shape.dim(0).value(): " << new_shape.dim(0).value() << std::endl;
  std::cout << "new_shape.dim(1).value(): " << new_shape.dim(1).value() << std::endl;

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&reshape, shape));

  ASSERT_EQ(2, shape.rank());
}