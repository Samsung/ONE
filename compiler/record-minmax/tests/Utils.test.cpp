/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Utils.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

using namespace record_minmax;

TEST(UtilsTest, num_elements)
{
  luci::CircleAdd node;
  node.rank(3);
  node.dim(0).set(1);
  node.dim(1).set(2);
  node.dim(2).set(3);
  node.dtype(loco::DataType::FLOAT32);

  EXPECT_EQ(6, numElements(&node));
}

TEST(UtilsTest, num_elements_NEG)
{
  luci::CircleAdd node;
  node.rank(3);
  node.dim(0).set(1);
  node.dim(1).set(2);
  node.dim(2).set(3);
  node.dtype(loco::DataType::FLOAT32);

  node.dim(0).unset();

  EXPECT_ANY_THROW(numElements(&node));
}

TEST(UtilsTest, get_tensor_size)
{
  luci::CircleAdd node;
  node.rank(3);
  node.dim(0).set(1);
  node.dim(1).set(2);
  node.dim(2).set(3);
  node.dtype(loco::DataType::FLOAT32);

  EXPECT_EQ(24, getTensorSize(&node));
}

TEST(UtilsTest, get_tensor_size_NEG)
{
  luci::CircleAdd node;
  node.rank(3);
  node.dim(0).set(1);
  node.dim(1).set(2);
  node.dim(2).set(3);
  node.dtype(loco::DataType::FLOAT32);

  node.dim(0).unset();

  EXPECT_ANY_THROW(getTensorSize(&node));
}

TEST(UtilsTest, check_input_dimension)
{
  luci::CircleInput node;
  node.rank(3);
  node.dim(0).set(1);
  node.dim(1).set(2);
  node.dim(2).set(3);
  node.dtype(loco::DataType::FLOAT32);

  EXPECT_NO_THROW(checkInputDimension(&node));
}

TEST(UtilsTest, check_input_dimension_unknown_dim_NEG)
{
  luci::CircleInput node;
  node.rank(3);
  node.dim(0).set(1);
  node.dim(1).set(2);
  node.dim(2).set(3);
  node.dtype(loco::DataType::FLOAT32);

  node.dim(0).unset();

  EXPECT_ANY_THROW(checkInputDimension(&node));
}

TEST(UtilsTest, check_input_dimension_zero_dim_NEG)
{
  luci::CircleInput node;
  node.rank(3);
  node.dim(0).set(1);
  node.dim(1).set(2);
  node.dim(2).set(0);
  node.dtype(loco::DataType::FLOAT32);

  EXPECT_ANY_THROW(checkInputDimension(&node));
}
