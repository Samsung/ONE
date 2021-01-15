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

#include <gtest/gtest.h>

#include "NodeFiller.h"

TEST(NodeFillerTest, simple_test)
{
  luci::CircleConst maxi_const;
  luci::CircleMinimum mini;
  luci::CircleMaximum maxi;
  maxi.x(&maxi_const);
  maxi.y(&mini);

  luci::CircleConst *x = nullptr;
  luci::CircleMinimum *y = nullptr;

  EXPECT_TRUE(luci::fill(&x, &y).with_commutative_args_of(&maxi));
  EXPECT_TRUE(x == &maxi_const);
  EXPECT_TRUE(y == &mini);

  x = nullptr;
  y = nullptr;

  EXPECT_TRUE(luci::fill(&y, &x).with_commutative_args_of(&maxi));
  EXPECT_TRUE(x == &maxi_const);
  EXPECT_TRUE(y == &mini);
}

TEST(NodeFillerTest, wrong_condition_NEG)
{
  luci::CircleConst add_const;
  luci::CircleMinimum mini;
  luci::CircleAdd add;
  add.x(&add_const);
  add.y(&mini);

  luci::CircleMul *x = nullptr;
  luci::CircleMinimum *y = nullptr;

  EXPECT_FALSE(luci::fill(&x, &y).with_commutative_args_of(&add));
  EXPECT_FALSE(luci::fill(&y, &x).with_commutative_args_of(&add));
}
