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

#include "loco/Service/TypeInference.h"

#include "GraphTestcase.h"

#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/TypeInference.h>

#include <vector>

#include <gtest/gtest.h>

// This test validates whether framework works as expected.
TEST(TypeInferenceTest, framework)
{
  // Create a sample network
  auto g = loco::make_graph();

  auto pull_node = g->nodes()->create<loco::Pull>();
  auto push_node = g->nodes()->create<loco::Push>();

  push_node->from(pull_node);

  // Create Graph Input & Output
  auto graph_input = g->inputs()->create();

  graph_input->name("input");
  loco::link(graph_input, pull_node);

  auto graph_output = g->outputs()->create();

  graph_output->name("output");
  loco::link(graph_output, push_node);

  // Mock-up Type Inference Rule
  struct SampleTypeInferenceRule final : public loco::TypeInferenceRule
  {
  public:
    SampleTypeInferenceRule(std::vector<const loco::Node *> *nodes) : _nodes{nodes}
    {
      // DO NOTHING
    }

  public:
    bool recognize(const loco::Dialect *) const final
    {
      // Accept all the dialects
      return true;
    }

    bool infer(const loco::Node *node, loco::DataType &dtype) const final
    {
      // Record the order of inference
      _nodes->emplace_back(node);

      if (_nodes->size() != 1)
      {
        return false;
      }

      // Annotate the first node as "U8"
      dtype = loco::DataType::U8;
      return true;
    }

  private:
    std::vector<const loco::Node *> *_nodes;
  };

  std::vector<const loco::Node *> nodes;

  SampleTypeInferenceRule rule{&nodes};

  loco::apply(&rule).to(g.get());

  ASSERT_EQ(2, nodes.size());        // Framework SHOULD visit all the nodes
  ASSERT_EQ(pull_node, nodes.at(0)); // Framework SHOULD visit "pull" before "push"
  ASSERT_EQ(push_node, nodes.at(1));

  // Framework SHOULD NOT make any annotation if "rule" returns FALSE
  ASSERT_TRUE(loco::dtype_known(pull_node));
  // Framework SHOULD make an annotation if "rule" returns TRUE
  ASSERT_EQ(loco::DataType::U8, loco::dtype_get(pull_node));
  ASSERT_FALSE(loco::dtype_known(push_node));
}

TEST(CanonicalTypeInferenceRuleTest, minimal)
{
  // Create a simple network
  auto g = loco::make_graph();

  auto pull_node = g->nodes()->create<loco::Pull>();

  pull_node->dtype(loco::DataType::U8);

  auto push_node = g->nodes()->create<loco::Push>();

  push_node->from(pull_node);

  auto graph_input = g->inputs()->create();

  graph_input->name("input");
  loco::link(graph_input, pull_node);

  auto graph_output = g->outputs()->create();

  graph_output->name("output");
  loco::link(graph_output, push_node);

  // Run Type Inference
  loco::CanonicalTypeInferenceRule rule;

  loco::apply(&rule).to(g.get());

  // Verify!
  ASSERT_TRUE(loco::dtype_known(push_node));
  ASSERT_EQ(loco::DataType::U8, loco::dtype_get(push_node));
}

TEST(CanonicalTypeInferenceRuleTest, relu6)
{
  // Create a simple Relu6 network
  auto g = loco::make_graph();

  auto pull_node = g->nodes()->create<loco::Pull>();

  pull_node->dtype(loco::DataType::FLOAT32);

  auto relu6_node = g->nodes()->create<loco::ReLU6>();

  relu6_node->input(pull_node);

  auto push_node = g->nodes()->create<loco::Push>();

  push_node->from(relu6_node);

  auto graph_input = g->inputs()->create();

  graph_input->name("input");
  loco::link(graph_input, pull_node);

  auto graph_output = g->outputs()->create();

  graph_output->name("output");
  loco::link(graph_output, push_node);

  // Run Type Inference
  loco::CanonicalTypeInferenceRule rule;

  loco::apply(&rule).to(g.get());

  // Verify!
  ASSERT_TRUE(loco::dtype_known(relu6_node));
  ASSERT_EQ(loco::DataType::FLOAT32, loco::dtype_get(relu6_node));
}

TEST(CanonicalTypeInferenceRuleTest, tensor_broadcast)
{
  // Create a sample network
  GraphTestcase<GraphCode::TensorBroadcast> testcase{1, 2};

  testcase.graph()->inputs()->at(0)->dtype(loco::DataType::U8);

  // Run Type Inference
  loco::CanonicalTypeInferenceRule rule;

  loco::apply(&rule).to(testcase.graph());

  // Verify!
  ASSERT_TRUE(loco::dtype_known(testcase.push_node));
  ASSERT_EQ(loco::DataType::U8, loco::dtype_get(testcase.push_node));
}

// mockup for MultiDialectTypeInferenceRule
// OpNode of a specific loco datatype (defined in template) will be used.
// And a Dialect for the OpNode and its inference rules are created.
#include <loco/IR/Dialect.h>

namespace
{

template <loco::DataType N> class TestDialect final : public loco::Dialect
{
public:
  static Dialect *get(void)
  {
    static TestDialect<N> d;
    return &d;
  }
};

template <loco::DataType N>
struct TestOpNode final : public loco::FixedArity<1>::Mixin<loco::Node>,
                          public loco::NodeMixin<loco::NodeTrait::DataType>
{
  void input(Node *node) { at(0)->node(node); }
  const loco::Dialect *dialect(void) const final { return TestDialect<N>::get(); }
  uint32_t opnum(void) const final { return static_cast<uint32_t>(N); }
};

template <loco::DataType N> struct TestTypeInferenceRule final : public loco::TypeInferenceRule
{
public:
  bool recognize(const loco::Dialect *d) const final { return (d == TestDialect<N>::get()); }

  bool infer(const loco::Node *node, loco::DataType &dtype) const final
  {
    assert(node->dialect() == TestDialect<N>::get());
    auto test_node = dynamic_cast<const TestOpNode<N> *>(node);
    assert(test_node != nullptr);

    dtype = N;
    return true;
  }
};

} // namespace

TEST(MultiDialectTypeInferenceRuleTest, test1)
{
  // Create a simple network : Pull - S4 - S8 - U4 - U8 - Push
  auto g = loco::make_graph();

  auto pull_node = g->nodes()->create<loco::Pull>();
  pull_node->dtype(loco::DataType::FLOAT32);

  auto s4_node = g->nodes()->create<TestOpNode<loco::DataType::S4>>();
  s4_node->input(pull_node);

  auto s8_node = g->nodes()->create<TestOpNode<loco::DataType::S8>>();
  s8_node->input(s4_node);

  auto u4_node = g->nodes()->create<TestOpNode<loco::DataType::U4>>();
  u4_node->input(s8_node);

  auto u8_node = g->nodes()->create<TestOpNode<loco::DataType::U8>>();
  u8_node->input(u4_node);

  auto push_node = g->nodes()->create<loco::Push>();
  push_node->from(u8_node);

  auto graph_input = g->inputs()->create();
  graph_input->name("input");
  loco::link(graph_input, pull_node);

  auto graph_output = g->outputs()->create();
  graph_output->name("output");
  loco::link(graph_output, push_node);

  // initially they don't have type info
  ASSERT_FALSE(loco::dtype_known(s4_node));
  ASSERT_FALSE(loco::dtype_known(s8_node));
  ASSERT_FALSE(loco::dtype_known(u4_node));
  ASSERT_FALSE(loco::dtype_known(u8_node));

  // Run Type Inference
  TestTypeInferenceRule<loco::DataType::U8> u8_rule;
  TestTypeInferenceRule<loco::DataType::S8> s8_rule;
  TestTypeInferenceRule<loco::DataType::S4> s4_rule;
  TestTypeInferenceRule<loco::DataType::U4> u4_rule;
  loco::CanonicalTypeInferenceRule canon_rule;

  loco::MultiDialectTypeInferenceRule rules;

  rules.bind(TestDialect<loco::DataType::S8>::get(), &s8_rule)
    .bind(TestDialect<loco::DataType::U8>::get(), &u8_rule)
    .bind(TestDialect<loco::DataType::S4>::get(), &s4_rule)
    .bind(TestDialect<loco::DataType::U4>::get(), &u4_rule)
    .bind(loco::CanonicalDialect::get(), &canon_rule);

  loco::apply(&rules).to(g.get());

  // Verify!
  ASSERT_TRUE(loco::dtype_known(s4_node));
  ASSERT_EQ(loco::DataType::S4, loco::dtype_get(s4_node));

  ASSERT_TRUE(loco::dtype_known(s8_node));
  ASSERT_EQ(loco::DataType::S8, loco::dtype_get(s8_node));

  ASSERT_TRUE(loco::dtype_known(u4_node));
  ASSERT_EQ(loco::DataType::U4, loco::dtype_get(u4_node));

  ASSERT_TRUE(loco::dtype_known(u8_node));
  ASSERT_EQ(loco::DataType::U8, loco::dtype_get(u8_node));
}
