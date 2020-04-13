/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleReshape.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleReshapeTest, constructor_P)
{
  luci::CircleReshape reshape;

  ASSERT_EQ(reshape.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(reshape.opcode(), luci::CircleOpcode::RESHAPE);

  ASSERT_EQ(reshape.tensor(), nullptr);
  ASSERT_EQ(reshape.shape(), nullptr);
  ASSERT_EQ(reshape.newShape()->rank(), 0);
}

TEST(CircleReshapeTest, alloc_new_shape_P)
{
  luci::CircleReshape reshape;

  reshape.newShape()->rank(2);
  ASSERT_EQ(reshape.newShape()->rank(), 2);

  reshape.newShape()->dim(0) = 0;
  reshape.newShape()->dim(1) = 1;

  auto &const_reshape = const_cast<const luci::CircleReshape &>(reshape);
  ASSERT_EQ(const_reshape.newShape()->dim(0), 0);
  ASSERT_EQ(const_reshape.newShape()->dim(1), 1);
}
