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

#include "nest/Ret.h"

#include <gtest/gtest.h>

namespace
{
struct DummyNode final : public nest::expr::Node
{
};
} // namespace

TEST(RET, ctor)
{
  nest::DomainID dom_id{0};
  nest::expr::Subscript sub{std::make_shared<DummyNode>()};

  nest::Ret ret{dom_id, sub};

  ASSERT_EQ(0, ret.id().value());
  ASSERT_EQ(1, ret.sub().rank());
}

TEST(RET, copy)
{
  nest::DomainID src_id{0};
  nest::expr::Subscript src_sub{std::make_shared<DummyNode>()};

  const nest::Ret src{src_id, src_sub};

  nest::DomainID dst_id{1};
  nest::expr::Subscript dst_sub{std::make_shared<DummyNode>(), std::make_shared<DummyNode>()};

  nest::Ret dst{dst_id, dst_sub};

  ASSERT_EQ(1, dst.id().value());
  ASSERT_EQ(2, dst.sub().rank());

  dst = src;

  ASSERT_EQ(0, dst.id().value());
  ASSERT_EQ(1, dst.sub().rank());
}
