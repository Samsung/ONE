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

#include "luci/Profile/CircleNodeID.h"
#include "luci/Profile/CircleNodeOrigin.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

TEST(LuciCircleNodeOrigin, simple_single_origin)
{
  auto g = loco::make_graph();
  auto add = g->nodes()->create<luci::CircleAdd>();

  ASSERT_FALSE(has_origin(add));

  auto origin = luci::single_origin(3, "add");
  add_origin(add, origin);

  ASSERT_TRUE(has_origin(add));

  auto sources = get_origin(add)->sources();
  ASSERT_EQ(1, sources.size());
  for (auto source : sources)
  {
    ASSERT_EQ(3, source->id());
    ASSERT_EQ(0, source->name().compare("add"));
  }
}

TEST(LuciCircleNodeOrigin, simple_composite_origin_with_initializer)
{
  auto g = loco::make_graph();
  auto mul = g->nodes()->create<luci::CircleMul>();

  ASSERT_FALSE(has_origin(mul));

  auto origin =
    luci::composite_origin({luci::single_origin(3, "add"), luci::single_origin(7, "sub")});
  add_origin(mul, origin);

  ASSERT_TRUE(has_origin(mul));

  bool add_origin_passed = false;
  bool sub_origin_passed = false;
  auto sources = get_origin(mul)->sources();
  ASSERT_EQ(2, sources.size());
  for (auto source : sources)
  {
    if (source->id() == 3 && source->name().compare("add") == 0)
      add_origin_passed = true;
    if (source->id() == 7 && source->name().compare("sub") == 0)
      sub_origin_passed = true;
  }

  ASSERT_EQ(true, add_origin_passed);
  ASSERT_EQ(true, sub_origin_passed);
}

TEST(LuciCircleNodeOrigin, simple_composite_origin_with_vector)
{
  auto g = loco::make_graph();
  auto mul = g->nodes()->create<luci::CircleMul>();

  ASSERT_FALSE(has_origin(mul));

  std::vector<std::shared_ptr<luci::CircleNodeOrigin>> vec;
  vec.push_back(luci::single_origin(3, "add"));
  vec.push_back(luci::single_origin(7, "sub"));
  auto origin = luci::composite_origin(vec);
  add_origin(mul, origin);

  ASSERT_TRUE(has_origin(mul));

  bool add_origin_passed = false;
  bool sub_origin_passed = false;
  auto sources = get_origin(mul)->sources();
  ASSERT_EQ(2, sources.size());
  for (auto source : sources)
  {
    if (source->id() == 3 && source->name().compare("add") == 0)
      add_origin_passed = true;
    if (source->id() == 7 && source->name().compare("sub") == 0)
      sub_origin_passed = true;
  }

  ASSERT_EQ(true, add_origin_passed);
  ASSERT_EQ(true, sub_origin_passed);
}

TEST(LuciCircleNodeOrigin, get_origin_NEG)
{
  auto g = loco::make_graph();
  auto add = g->nodes()->create<luci::CircleAdd>();

  ASSERT_FALSE(has_origin(add));

  ASSERT_ANY_THROW(get_origin(add));
}

TEST(LuciCircleNodeOrigin, composite_origin_empty_ctor_NEG)
{
  ASSERT_ANY_THROW(luci::composite_origin({}));
}

TEST(LuciCircleNodeOrigin, composite_origin_null_ctor_NEG)
{
  ASSERT_ANY_THROW(luci::composite_origin({nullptr}));
}
