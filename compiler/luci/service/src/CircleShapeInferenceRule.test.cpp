/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TestGraph.h"
#include "luci/Service/CircleShapeInferenceRule.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleDialect.h>

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/CanonicalShapeInferenceRule.h>
#include <loco/Service/MultiDialectShapeInferenceRule.h>

#include <gtest/gtest.h>

#include <memory>

namespace
{

bool shape_pass(loco::Graph *g)
{
  loco::CanonicalShapeInferenceRule canonical_rule;
  luci::CircleShapeInferenceRule circle_rule;
  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
      .bind(luci::CircleDialect::get(), &circle_rule);

  return loco::apply(&rules).to(g);
}

} // namespace

TEST(CircleShapeInferenceRuleTest, minimal_with_CircleRelu)
{
  // Create a simple network
  luci::test::TestGraph graph;
  auto tfl_node = graph.append<luci::CircleRelu>(graph.pull);
  graph.complete(tfl_node);

  // set shape
  {
    graph.pull->rank(2);
    graph.pull->dim(0) = 3;
    graph.pull->dim(1) = 4;
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(tfl_node));

  // shape inference
  luci::CircleShapeInferenceRule tfl_rule;
  loco::CanonicalShapeInferenceRule canonical_rule;
  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
      .bind(luci::CircleDialect::get(), &tfl_rule);

  loco::apply(&rules).to(graph.g.get());

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(tfl_node));
    ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(tfl_node).domain());

    auto shape = loco::shape_get(tfl_node).as<loco::TensorShape>();
    ASSERT_EQ(2, shape.rank());
    ASSERT_EQ(3, shape.dim(0));
    ASSERT_EQ(4, shape.dim(1));
  }
}

// based on the case shown in
// https://www.corvil.com/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow
TEST(CircleShapeInferenceRuleTest, avgpool2d_valid)
{
  luci::test::TestGraph graph;
  auto tfl_node = graph.append<luci::CircleAveragePool2D>(graph.pull);
  graph.complete();

  auto pull = graph.pull;
  {
    pull->shape({1, 4, 3, 1});
  }
  // setting CircleAveragePool2D
  {
    tfl_node->filter()->h(2);
    tfl_node->filter()->w(2);
    tfl_node->stride()->h(2);
    tfl_node->stride()->w(2);
    tfl_node->fusedActivationFunction(luci::FusedActFunc::NONE);
    tfl_node->padding(luci::Padding::VALID);
  }
  ASSERT_FALSE(loco::shape_known(tfl_node));

  // shape inference
  luci::CircleShapeInferenceRule tfl_rule;
  loco::CanonicalShapeInferenceRule canonical_rule;
  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
      .bind(luci::CircleDialect::get(), &tfl_rule);

  loco::apply(&rules).to(graph.g.get());

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(tfl_node));
    ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(tfl_node).domain());

    auto shape = loco::shape_get(tfl_node).as<loco::TensorShape>();
    ASSERT_EQ(4, shape.rank());
    ASSERT_EQ(1, shape.dim(0).value());
    ASSERT_EQ(2, shape.dim(1).value());
    ASSERT_EQ(1, shape.dim(2).value());
    ASSERT_EQ(1, shape.dim(3).value());
  }
}

TEST(CircleShapeInferenceRuleTest, avgpool2d_same)
{
  luci::test::TestGraph graph;
  auto tfl_node = graph.append<luci::CircleAveragePool2D>(graph.pull);
  graph.complete();

  auto pull = graph.pull;
  {
    pull->shape({1, 4, 3, 1});
  }

  // setting CircleAveragePool2D
  {
    tfl_node->filter()->h(2);
    tfl_node->filter()->w(2);
    tfl_node->stride()->h(2);
    tfl_node->stride()->w(2);
    tfl_node->fusedActivationFunction(luci::FusedActFunc::NONE);
    tfl_node->padding(luci::Padding::SAME);
  }

  ASSERT_FALSE(loco::shape_known(tfl_node));

  // shape inference
  shape_pass(graph.g.get());

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(tfl_node));
    ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(tfl_node).domain());

    auto shape = loco::shape_get(tfl_node).as<loco::TensorShape>();
    ASSERT_EQ(4, shape.rank());
    ASSERT_EQ(1, shape.dim(0).value());
    ASSERT_EQ(2, shape.dim(1).value());
    ASSERT_EQ(2, shape.dim(2).value());
    ASSERT_EQ(1, shape.dim(3).value());
  }
}

/**
 * @note Function to test: Shape inference of two different input shapes
 *
 *       Rank expansion to higher input side
 *          x(2,1,5) + y(3,5) --> x(2,1,5) + y(1,3,5)
 *       Do output shape inference like numpy
 *          x(2,1,5) + y(1,3,5) --> output(2,3,5)
 *       For each axis, dim value should be same OR one of them should be 1
 */
TEST(CircleShapeInferenceRuleTest, TFAdd_shapeinf_different)
{
  auto g = loco::make_graph();

  auto x_node = g->nodes()->create<loco::Pull>();
  {
    x_node->rank(3);
    x_node->dim(0) = 2;
    x_node->dim(1) = 1;
    x_node->dim(2) = 5;
  }
  auto y_node = g->nodes()->create<loco::Pull>();
  {
    y_node->rank(2);
    y_node->dim(0) = 3;
    y_node->dim(1) = 5;
  }
  auto tfl_node = g->nodes()->create<luci::CircleAdd>();
  {
    tfl_node->x(x_node);
    tfl_node->y(y_node);
  }
  auto push_node = g->nodes()->create<loco::Push>();
  {
    push_node->from(tfl_node);
  }

  auto x_input = g->inputs()->create();
  {
    x_input->name("x");
    loco::link(x_input, x_node);
  }
  auto y_input = g->inputs()->create();
  {
    y_input->name("y");
    loco::link(y_input, y_node);
  }
  auto output = g->outputs()->create();
  {
    output->name("output");
    loco::link(output, push_node);
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(tfl_node));

  // shape inference
  while (shape_pass(g.get()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(tfl_node));
    ASSERT_EQ(loco::Domain::Tensor, loco::shape_get(tfl_node).domain());

    auto shape = loco::shape_get(tfl_node).as<loco::TensorShape>();
    ASSERT_EQ(3, shape.rank());
    ASSERT_EQ(2, shape.dim(0));
    ASSERT_EQ(3, shape.dim(1));
    ASSERT_EQ(5, shape.dim(2));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleTranspose_simple)
{
  luci::test::ExampleGraph<luci::test::ExampleGraphType::CircleTranspose> g;

  g.pull->rank(3);
  g.pull->dim(0) = 3;
  g.pull->dim(1) = 8;
  g.pull->dim(2) = 1;

  g.const_perm->dtype(loco::DataType::S32);
  g.const_perm->rank(1);
  g.const_perm->dim(0) = 3;
  g.const_perm->size<loco::DataType::S32>(3);
  g.const_perm->at<loco::DataType::S32>(0) = 1;
  g.const_perm->at<loco::DataType::S32>(1) = 2;
  g.const_perm->at<loco::DataType::S32>(2) = 0;

  // pre-check
  ASSERT_FALSE(loco::shape_known(g.transpose_node));

  // shape inference
  while (shape_pass(g.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(g.transpose_node));

    auto shape = loco::shape_get(g.transpose_node).as<loco::TensorShape>();
    ASSERT_EQ(3, shape.rank());
    ASSERT_EQ(8, shape.dim(0));
    ASSERT_EQ(1, shape.dim(1));
    ASSERT_EQ(3, shape.dim(2));
  }
}
