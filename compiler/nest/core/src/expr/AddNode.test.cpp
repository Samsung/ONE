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

#include "nest/expr/AddNode.h"

#include <memory>

#include <gtest/gtest.h>

namespace
{
struct DummyNode final : public nest::expr::Node
{
};
} // namespace

TEST(ADD_NODE, cast)
{
  auto left = std::make_shared<DummyNode>();
  auto right = std::make_shared<DummyNode>();

  auto derived = std::make_shared<nest::expr::AddNode>(left, right);
  std::shared_ptr<nest::expr::Node> base = derived;

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.get(), base->asAdd());

  ASSERT_EQ(left.get(), derived->lhs().get());
  ASSERT_EQ(right.get(), derived->rhs().get());
}
