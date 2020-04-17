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

#include "luci/IR/Module.h"

#include <gtest/gtest.h>

TEST(ModuleTest, consturctor)
{
  auto gs = luci::make_module();

  GTEST_SUCCEED();
}

TEST(ModuleTest, add)
{
  auto m = luci::make_module();
  auto g = loco::make_graph();
  auto g_ptr = g.get();

  m->add(std::move(g));

  ASSERT_EQ(m->graph(), g_ptr);
  ASSERT_EQ(m->graph(0), g_ptr);
}

TEST(ModuleTest, add_more)
{
  auto m = luci::make_module();
  auto g1 = loco::make_graph();
  auto g2 = loco::make_graph();
  auto g3 = loco::make_graph();
  auto g1_ptr = g1.get();
  auto g2_ptr = g2.get();
  auto g3_ptr = g3.get();

  m->add(std::move(g1));
  m->add(std::move(g2));
  m->add(std::move(g3));

  ASSERT_EQ(m->size(), 3);
  ASSERT_EQ(m->graph(), g1_ptr);
  ASSERT_EQ(m->graph(0), g1_ptr);
  ASSERT_EQ(m->graph(1), g2_ptr);
  ASSERT_EQ(m->graph(2), g3_ptr);
}

TEST(ModuleTest, add_nullptr_NEG)
{
  auto m = luci::make_module();

  EXPECT_THROW(m->add(nullptr), std::invalid_argument);
}

TEST(ModuleTest, graph_index_overflow_NEG)
{
  auto m = luci::make_module();

  EXPECT_ANY_THROW(m->graph(100));
}
