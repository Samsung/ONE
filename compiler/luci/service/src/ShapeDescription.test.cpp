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

#include "luci/Service/ShapeDescription.h"

#include <gtest/gtest.h>

TEST(ShapeDescriptionTest, TensorShape)
{
  loco::TensorShape tensor_shape{1, 2, 3, 4};
  loco::NodeShape node_shape(tensor_shape);

  auto sd = luci::to_shape_description(node_shape);

  ASSERT_EQ(4, sd._dims.size());
  ASSERT_EQ(1, sd._dims.at(0));
  ASSERT_TRUE(sd._rank_known);
}

TEST(ShapeDescriptionTest, BiasShape_NEG)
{
  loco::BiasShape bias_shape;
  bias_shape.length() = 1;
  loco::NodeShape node_shape(bias_shape);

  EXPECT_THROW(luci::to_shape_description(node_shape), std::exception);
}
