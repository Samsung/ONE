/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loco/IR/Dialect.h"

#include <memory>

#include <gtest/gtest.h>

TEST(DialectTest, service)
{
  struct S0 final : public loco::DialectService
  {
  };
  struct S1 final : public loco::DialectService
  {
  };

  struct MockDialect final : public loco::Dialect
  {
    MockDialect() { service<S1>(std::make_unique<S1>()); }
  };

  MockDialect dialect;

  ASSERT_EQ(nullptr, dialect.service<S0>());
  ASSERT_NE(dialect.service<S1>(), nullptr);
}
