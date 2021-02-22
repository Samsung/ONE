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

#include "locop/FormattedGraph.h"
#include "ExampleGraph.h"

#include <memory>

#include <gtest/gtest.h>

TEST(LinearV1FormatterTest, simple)
{
  auto bundle = make_bundle<PullPush>();
  auto g = bundle->graph();

  // TODO Validate the output (when the implementation becomes stable)
  std::cout << locop::fmt<locop::LinearV1>(g) << std::endl;

  SUCCEED();
}

TEST(LinearV1FormatterTest, user_defined_node_summary_builder)
{
  struct MyAnnotation final : public loco::NodeAnnotation
  {
    // DO NOTHING
  };

  auto bundle = make_bundle<PullPush>();
  auto g = bundle->graph();
  {
    bundle->push->annot(std::make_unique<MyAnnotation>());
  }

  struct MyBuilder final : public locop::NodeSummaryBuilder
  {
    bool build(const loco::Node *node, locop::NodeSummary &s) const final
    {
      s.opname("my.op");
      if (node->annot<MyAnnotation>())
      {
        s.comments().append("annotated");
      }
      s.state(locop::NodeSummary::State::PartiallyKnown);
      return true;
    }
  };

  struct MyFactory final : public locop::NodeSummaryBuilderFactory
  {
    std::unique_ptr<locop::NodeSummaryBuilder> create(const locop::SymbolTable *) const final
    {
      return std::make_unique<MyBuilder>();
    }
  };

  std::cout << locop::fmt<locop::LinearV1>(g).with(std::make_unique<MyFactory>()) << std::endl;

  // TODO Check whether MyBuilder actually sees all the nodes in a graph
  SUCCEED();
}

// This test shows how to compose two node summary builders.
TEST(LinearV1FormatterTest, node_summary_builder_composition)
{
  struct MyNode : public loco::FixedArity<0>::Mixin<loco::Node>
  {
    uint32_t opnum(void) const final { return 0; }
    const loco::Dialect *dialect(void) const final { return nullptr; };
  };

  auto g = loco::make_graph();
  {
    auto user = g->nodes()->create<MyNode>();

    auto push = g->nodes()->create<loco::Push>();

    push->from(user);
  }

  // TODO Reuse MyBuilder above
  struct MyBuilder final : public locop::NodeSummaryBuilder
  {
    bool build(const loco::Node *node, locop::NodeSummary &s) const final
    {
      s.opname("my.op");
      s.state(locop::NodeSummary::State::PartiallyKnown);
      return true;
    }
  };

  class CompositeBuilder final : public locop::NodeSummaryBuilder
  {
  public:
    CompositeBuilder(const locop::SymbolTable *tbl) : _tbl{tbl}
    {
      // DO NOTHING
    }

  public:
    bool build(const loco::Node *node, locop::NodeSummary &s) const final
    {
      if (locop::CanonicalNodeSummaryBuilder(_tbl).build(node, s))
      {
        return true;
      }

      if (MyBuilder().build(node, s))
      {
        return true;
      }

      return false;
    }

  private:
    const locop::SymbolTable *_tbl;
  };

  struct MyFactory final : public locop::NodeSummaryBuilderFactory
  {
    std::unique_ptr<locop::NodeSummaryBuilder> create(const locop::SymbolTable *tbl) const final
    {
      return std::make_unique<CompositeBuilder>(tbl);
    }
  };

  std::cout << locop::fmt<locop::LinearV1>(g).with(std::make_unique<MyFactory>()) << std::endl;

  // TODO Check whether MyBuilder actually sees all the nodes in a graph
  SUCCEED();
}
