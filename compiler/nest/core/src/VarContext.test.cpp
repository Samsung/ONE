/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nest/VarContext.h"

#include <gtest/gtest.h>

TEST(VAR_CONTEXT, make)
{
  nest::VarContext ctx;

  auto var_0 = ctx.make();
  auto var_1 = ctx.make();

  ASSERT_FALSE(var_0.id() == var_1.id());
}

TEST(VAR_CONTEXT, count)
{
  nest::VarContext ctx;

  ASSERT_EQ(ctx.count(), 0);

  auto var_0 = ctx.make();

  ASSERT_EQ(ctx.count(), 1);

  auto var_1 = ctx.make();

  ASSERT_EQ(ctx.count(), 2);
}

TEST(VAR_CONTEXT, bound_one)
{
  nest::VarContext ctx;

  auto var_0 = ctx.make();

  ASSERT_EQ(ctx.bound(var_0).min(), 0);
  ASSERT_EQ(ctx.bound(var_0).max(), 0);

  ctx.bound(var_0) = nest::Bound{-3, 5};

  ASSERT_EQ(ctx.bound(var_0).min(), -3);
  ASSERT_EQ(ctx.bound(var_0).max(), 5);
}

TEST(VAR_CONTEXT, bound_independent)
{
  nest::VarContext ctx;

  auto var_0 = ctx.make();

  ASSERT_EQ(ctx.bound(var_0).min(), 0);
  ASSERT_EQ(ctx.bound(var_0).max(), 0);

  auto var_1 = ctx.make();

  ASSERT_EQ(ctx.bound(var_1).min(), 0);
  ASSERT_EQ(ctx.bound(var_1).max(), 0);

  ctx.bound(var_0) = nest::Bound{-3, 5};

  ASSERT_EQ(ctx.bound(var_0).min(), -3);
  ASSERT_EQ(ctx.bound(var_0).max(), 5);

  ASSERT_EQ(ctx.bound(var_1).min(), 0);
  ASSERT_EQ(ctx.bound(var_1).max(), 0);
}
