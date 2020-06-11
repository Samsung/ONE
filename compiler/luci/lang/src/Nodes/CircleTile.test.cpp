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

#include "luci/IR/Nodes/CircleTile.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleTileTest, constructor)
{
  luci::CircleTile tile_node;

  ASSERT_EQ(luci::CircleDialect::get(), tile_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::TILE, tile_node.opcode());

  ASSERT_EQ(nullptr, tile_node.input());
  ASSERT_EQ(nullptr, tile_node.multiples());
}

TEST(CircleTileTest, input_NEG)
{
  luci::CircleTile tile_node;
  luci::CircleTile node;

  tile_node.input(&node);
  tile_node.multiples(&node);
  ASSERT_NE(nullptr, tile_node.input());
  ASSERT_NE(nullptr, tile_node.multiples());

  tile_node.input(nullptr);
  tile_node.multiples(nullptr);
  ASSERT_EQ(nullptr, tile_node.input());
  ASSERT_EQ(nullptr, tile_node.multiples());
}

TEST(CircleTileTest, arity_NEG)
{
  luci::CircleTile tile_node;

  ASSERT_NO_THROW(tile_node.arg(1));
  ASSERT_THROW(tile_node.arg(2), std::out_of_range);
}

TEST(CircleTileTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleTile tile_node;

  TestVisitor tv;
  ASSERT_THROW(tile_node.accept(&tv), std::exception);
}

TEST(CircleTileTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleTile tile_node;

  TestVisitor tv;
  ASSERT_THROW(tile_node.accept(&tv), std::exception);
}
