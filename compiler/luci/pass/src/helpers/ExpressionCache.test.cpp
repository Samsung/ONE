/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExpressionCache.h"

using namespace luci::pass;

TEST(ExpressionCacheTest, simple_test)
{
  luci::CircleInput in;
  luci::CircleQuantize node;
  node.input(&in);

  auto expr = Expression::build(&node);

  ExpressionCache cache;

  cache.put(expr, &node);

  EXPECT_NE(nullptr, cache.get(expr));
}

TEST(ExpressionCacheTest, null_expr_NEG) { EXPECT_ANY_THROW(Expression::build(nullptr)); }
