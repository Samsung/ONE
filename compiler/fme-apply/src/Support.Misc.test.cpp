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

#include "Support.Misc.h"

#include <gtest/gtest.h>

using namespace fme_apply;

TEST(SupportMiscTest, copy_shape)
{
  loco::Graph g;

  auto from = g.nodes()->create<luci::CircleConv2D>();
  from->shape({1, 2, 3, 4});

  auto to = g.nodes()->create<luci::CircleCustom>(2, 1);
  to->shape({1, 2, 3, 4});

  EXPECT_NO_THROW(copy_shape(from, to));

  EXPECT_EQ(from->rank(), to->rank());
  for (uint32_t i = 0; i < from->rank(); i++)
  {
    EXPECT_EQ(from->dim(i).value(), to->dim(i).value());
  }
}

TEST(SupportMiscTest, copy_shape_from_null_NEG)
{
  loco::Graph g;

  auto to = g.nodes()->create<luci::CircleCustom>(2, 1);
  to->shape({1, 2, 3, 4});

  EXPECT_ANY_THROW(copy_shape(nullptr, to));
}

TEST(SupportMiscTest, copy_shape_to_null_NEG)
{
  loco::Graph g;

  auto from = g.nodes()->create<luci::CircleConv2D>();
  from->shape({1, 2, 3, 4});

  EXPECT_ANY_THROW(copy_shape(from, nullptr));
}

TEST(SupportMiscTest, get_input_simple)
{
  loco::Graph g;

  auto input_node = g.nodes()->create<luci::CircleCustom>(2, 1);

  auto node = g.nodes()->create<luci::CircleConv2D>();
  node->shape({1, 2, 3, 4});
  node->input(input_node);

  auto ret = get_input(node);
  EXPECT_EQ(ret, input_node);
}

TEST(SupportMiscTest, set_input_simple)
{
  loco::Graph g;

  auto input_node = g.nodes()->create<luci::CircleCustom>(2, 1);

  auto node = g.nodes()->create<luci::CircleConv2D>();
  node->shape({1, 2, 3, 4});

  set_input(node, input_node);
  EXPECT_EQ(node->input(), input_node);
}

TEST(SupportMiscTest, find_arg_with_name_simple)
{
  loco::Graph g;

  auto input_node = g.nodes()->create<luci::CircleCustom>(2, 1);
  input_node->name("input_node");

  auto node = g.nodes()->create<luci::CircleConv2D>();
  node->shape({1, 2, 3, 4});
  node->input(input_node);

  auto ret = find_arg_with_name(node, "input_node");

  EXPECT_EQ(ret, input_node);
}
