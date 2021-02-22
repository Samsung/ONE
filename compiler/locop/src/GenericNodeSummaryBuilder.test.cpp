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

#include "locop/GenericNodeSummaryBuilder.h"
#include "locop/FormattedGraph.h"

#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

TEST(GenericNodeSummaryBuilderTest, simple)
{
  struct MockDialect final : public loco::Dialect
  {
    static Dialect *get(void)
    {
      static MockDialect d;
      return &d;
    }
  };

  struct MockNode : public loco::FixedArity<0>::Mixin<loco::Node>
  {
    const loco::Dialect *dialect(void) const final { return MockDialect::get(); };
    uint32_t opnum(void) const final { return 0; }
  };

  struct MockFactory final : public locop::NodeSummaryBuilderFactory
  {
    std::unique_ptr<locop::NodeSummaryBuilder> create(const locop::SymbolTable *tbl) const final
    {
      return std::make_unique<locop::GenericNodeSummaryBuilder>(tbl);
    }
  };

  auto g = loco::make_graph();

  g->nodes()->create<MockNode>();

  std::cout << locop::fmt<locop::LinearV1>(g).with(std::make_unique<MockFactory>()) << std::endl;

  SUCCEED();
}
