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

#include "nest/Expr.h"

#include <memory>

#include <gtest/gtest.h>

namespace
{
struct DummyNode final : public nest::expr::Node
{
};
} // namespace

TEST(EXPR, operator_sum)
{
  auto left = std::make_shared<DummyNode>();
  auto right = std::make_shared<DummyNode>();

  auto expr = left + right;

  ASSERT_NE(expr->asAdd(), nullptr);

  auto add = expr->asAdd();

  ASSERT_EQ(left.get(), add->lhs().get());
  ASSERT_EQ(right.get(), add->rhs().get());
}

TEST(EXPR, operator_mul)
{
  auto left = std::make_shared<DummyNode>();
  auto right = std::make_shared<DummyNode>();

  auto expr = left * right;

  ASSERT_NE(expr->asMul(), nullptr);

  auto add = expr->asMul();

  ASSERT_EQ(left.get(), add->lhs().get());
  ASSERT_EQ(right.get(), add->rhs().get());
}
