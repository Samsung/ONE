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

#include "nest/stmt/PushNode.h"

#include <memory>

#include <gtest/gtest.h>

namespace
{
struct DummyExprNode final : public nest::expr::Node
{
};
} // namespace

TEST(STMT_PUSH_NODE, cast)
{
  auto derived = std::make_shared<nest::stmt::PushNode>(std::make_shared<DummyExprNode>());
  std::shared_ptr<nest::stmt::Node> base = derived;

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.get(), base->asPush());
}
