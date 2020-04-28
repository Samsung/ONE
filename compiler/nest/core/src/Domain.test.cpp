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

#include "nest/Domain.h"

#include <gtest/gtest.h>

namespace
{
namespace expr
{
struct DummyNode final : public nest::expr::Node
{
};
} // namespace expr
} // namespace

// NOTE Build failed when DOMAIN is used instead of _DOMAIN
TEST(_DOMAIN, base_usecase)
{
  nest::DomainID dom_id{0};
  nest::Domain dom{dom_id};

  nest::Closure clo = dom(std::make_shared<::expr::DummyNode>());

  ASSERT_EQ(dom_id, clo.id());
  ASSERT_EQ(1, clo.sub().rank());
}
