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

#include "loco/Service/CanonicalShapeInferenceRule.h"
#include "loco/Service/MultiDialectShapeInferenceRule.h"
#include "loco/Service/ShapeInference.h"

#include <loco/IR/Dialect.h>
#include <loco/IR/CanonicalDialect.h>

#include <gtest/gtest.h>

#include <cassert>
#include <vector>

// mockup for MultiDialectShapeInferenceRule
// Each class is dedicated for handling shape { D1, D2 } and D1, D2 are declared as a template
namespace
{

template <uint32_t D1, uint32_t D2> class TestDialect final : public loco::Dialect
{
public:
  static Dialect *get(void)
  {
    static TestDialect<D1, D2> d;
    return &d;
  }
};

template <uint32_t D1, uint32_t D2>
struct TestOpNode final : public loco::FixedArity<1>::Mixin<loco::Node>,
                          public loco::NodeMixin<loco::NodeTrait::TensorShape>
{
  void input(Node *node) { at(0)->node(node); }
  const loco::Dialect *dialect(void) const final { return TestDialect<D1, D2>::get(); }
  uint32_t opnum(void) const final { return static_cast<uint32_t>(D1); /* not used */ }
};

template <uint32_t D1, uint32_t D2>
struct TestShapeInferenceRule final : public loco::ShapeInferenceRule
{
public:
  bool recognize(const loco::Dialect *d) const final { return (d == TestDialect<D1, D2>::get()); }

  bool infer(const loco::Node *node, loco::NodeShape &node_shape) const final
  {
    assert(recognize(node->dialect()));
    auto test_node = dynamic_cast<const TestOpNode<D1, D2> *>(node);
    assert(test_node != nullptr);

    loco::TensorShape ts;
    {
      ts.rank(2);
      ts.dim(0) = D1;
      ts.dim(1) = D2; // making shape : { D1, D2 }
    }

    node_shape.set(ts);

    return true;
  }
};

} // namespace

TEST(MultiDialectShapeInferenceRuleTest, test1)
{
  // Create a simple network : Pull ------- t23<2,3> ------------ t45<4,5> ---------- Push
  //                                  TensorShape({2, 3})    TensorShape({4, 5})
  auto g = loco::make_graph();

  auto pull_node = g->nodes()->create<loco::Pull>();
  auto t23_node = g->nodes()->create<TestOpNode<2, 3>>();
  auto t45_node = g->nodes()->create<TestOpNode<4, 5>>();
  auto push_node = g->nodes()->create<loco::Push>();

  t23_node->input(pull_node);
  t45_node->input(t23_node);
  push_node->from(t45_node);

  auto graph_input = g->inputs()->create();
  graph_input->name("input");
  loco::link(graph_input, pull_node);

  auto graph_output = g->outputs()->create();
  graph_output->name("output");
  loco::link(graph_output, push_node);

  // initially they don't have shape info
  ASSERT_FALSE(loco::shape_known(t23_node));
  ASSERT_FALSE(loco::shape_known(t45_node));

  // Run Type Inference
  loco::CanonicalShapeInferenceRule canonical_rule;
  TestShapeInferenceRule<2, 3> t23_rule;
  TestShapeInferenceRule<4, 5> t45_rule;

  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
    .bind(TestDialect<2, 3>::get(), &t23_rule)
    .bind(TestDialect<4, 5>::get(), &t45_rule);

  loco::apply(&rules).to(g.get());

  // Verify!
  ASSERT_TRUE(loco::shape_known(t23_node));
  auto t23_shape = loco::shape_get(t23_node);
  ASSERT_EQ(loco::Domain::Tensor, t23_shape.domain());
  ASSERT_EQ(2, t23_shape.as<loco::TensorShape>().rank());
  ASSERT_EQ(2, t23_shape.as<loco::TensorShape>().dim(0));
  ASSERT_EQ(3, t23_shape.as<loco::TensorShape>().dim(1));

  ASSERT_TRUE(loco::shape_known(t45_node));
  auto t45_shape = loco::shape_get(t45_node);
  ASSERT_EQ(loco::Domain::Tensor, t45_shape.domain());
  ASSERT_EQ(2, t45_shape.as<loco::TensorShape>().rank());
  ASSERT_EQ(4, t45_shape.as<loco::TensorShape>().dim(0));
  ASSERT_EQ(5, t45_shape.as<loco::TensorShape>().dim(1));
}
