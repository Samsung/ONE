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

#include "TestGraph.h"

#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLDialect.h"
#include "Dialect/Service/TFLShapeInferenceRule.h"

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/CanonicalShapeInferenceRule.h>
#include <loco/Service/MultiDialectShapeInferenceRule.h>

#include <stdex/Memory.h>

#include <gtest/gtest.h>

TEST(TFLShapeInferenceRuleTest, minimal_with_TFLRelu)
{
  // Create a simple network
  exo::test::TestGraph graph;
  auto tfl_node = graph.append<locoex::TFLRelu>(graph.pull);
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
  locoex::TFLShapeInferenceRule tfl_rule;
  loco::CanonicalShapeInferenceRule canonical_rule;
  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
    .bind(locoex::TFLDialect::get(), &tfl_rule);

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
TEST(TFLShapeInferenceRuleTest, avgpool2d_valid)
{
  exo::test::TestGraph graph;
  auto tfl_node = graph.append<locoex::TFLAveragePool2D>(graph.pull);
  graph.complete();

  auto pull = graph.pull;
  {
    pull->shape({1, 4, 3, 1});
  }
  // setting TFLAveragePool2D
  {
    tfl_node->filter()->h(2);
    tfl_node->filter()->w(2);
    tfl_node->stride()->h(2);
    tfl_node->stride()->w(2);
    tfl_node->fusedActivationFunction(locoex::FusedActFunc::NONE);
    tfl_node->padding(locoex::Padding::VALID);
  }
  ASSERT_FALSE(loco::shape_known(tfl_node));

  // shape inference
  locoex::TFLShapeInferenceRule tfl_rule;
  loco::CanonicalShapeInferenceRule canonical_rule;
  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
    .bind(locoex::TFLDialect::get(), &tfl_rule);

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

TEST(TFLShapeInferenceRuleTest, avgpool2d_same)
{
  exo::test::TestGraph graph;
  auto tfl_node = graph.append<locoex::TFLAveragePool2D>(graph.pull);
  graph.complete();

  auto pull = graph.pull;
  {
    pull->shape({1, 4, 3, 1});
  }

  // setting TFLAveragePool2D
  {
    tfl_node->filter()->h(2);
    tfl_node->filter()->w(2);
    tfl_node->stride()->h(2);
    tfl_node->stride()->w(2);
    tfl_node->fusedActivationFunction(locoex::FusedActFunc::NONE);
    tfl_node->padding(locoex::Padding::SAME);
  }

  ASSERT_FALSE(loco::shape_known(tfl_node));

  // shape inference
  locoex::TFLShapeInferenceRule tfl_rule;
  loco::CanonicalShapeInferenceRule canonical_rule;
  loco::MultiDialectShapeInferenceRule rules;

  rules.bind(loco::CanonicalDialect::get(), &canonical_rule)
    .bind(locoex::TFLDialect::get(), &tfl_rule);

  loco::apply(&rules).to(graph.g.get());

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
TEST(TFLShapeInferenceRuleTest, TFAdd_shapeinf_different)
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
  auto tfl_node = g->nodes()->create<locoex::TFLAdd>();
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

  exo::ShapeInferencePass pass;
  while (pass.run(g.get()) == true)
  {
    ;
  }

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

TEST(TFLShapeInferenceRuleTest, TFLTranspose_simple)
{
  exo::test::ExampleGraph<exo::test::ExampleGraphType::TFLTranspose> g;

  g.pull->rank(4);
  g.pull->dim(0) = 10;
  g.pull->dim(1) = 20;
  g.pull->dim(2) = 30;
  g.pull->dim(3) = 40;

  g.const_perm->dtype(loco::DataType::S32);
  g.const_perm->rank(1);
  g.const_perm->dim(0) = 4;
  g.const_perm->size<loco::DataType::S32>(4);
  g.const_perm->at<loco::DataType::S32>(0) = 2;
  g.const_perm->at<loco::DataType::S32>(1) = 3;
  g.const_perm->at<loco::DataType::S32>(2) = 0;
  g.const_perm->at<loco::DataType::S32>(3) = 1;

  // pre-check
  ASSERT_FALSE(loco::shape_known(g.tfl_transpose));

  exo::ShapeInferencePass pass;
  while (pass.run(g.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(g.tfl_transpose));

    auto shape = loco::shape_get(g.tfl_transpose).as<loco::TensorShape>();
    ASSERT_EQ(4, shape.rank());
    ASSERT_EQ(30, shape.dim(0));
    ASSERT_EQ(40, shape.dim(1));
    ASSERT_EQ(10, shape.dim(2));
    ASSERT_EQ(20, shape.dim(3));
  }
}
