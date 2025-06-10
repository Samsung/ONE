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

#include "Support.Cast.h"

#include <gtest/gtest.h>

using namespace fme_apply;

TEST(SupportCastTest, to_scale_simple)
{
  loco::Graph g;
  auto node = g.nodes()->create<luci::CircleCustom>(2, 1);
  node->custom_code("scale");

  EXPECT_EQ(node, to_scale(g.nodes()->at(0)));
}

TEST(SupportCastTest, to_scale_null_NEG) { EXPECT_EQ(nullptr, to_scale(nullptr)); }

TEST(SupportCastTest, to_scale_wrong_code_NEG)
{
  loco::Graph g;
  auto node = g.nodes()->create<luci::CircleCustom>(2, 1);
  node->custom_code("wrong");

  EXPECT_EQ(nullptr, to_scale(g.nodes()->at(0)));
}
