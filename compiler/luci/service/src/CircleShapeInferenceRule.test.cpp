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
#include "CircleShapeInferenceHelper.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleDialect.h>
#include <luci/Service/CircleShapeInference.h>

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/CanonicalShapeInferenceRule.h>
#include <loco/Service/MultiDialectShapeInferenceRule.h>

#include <oops/InternalExn.h>

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

bool is_same_shape(luci::CircleNode *node, loco::TensorShape shape)
{
  if (node->shape_status() != luci::ShapeStatus::VALID)
    return false;

  if (node->rank() != shape.rank())
    return false;

  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    if (node->dim(i).known() != shape.dim(i).known())
      return false;

    if (node->dim(i).value() != shape.dim(i).value())
      return false;
  }

  return true;
}

// NOTE This function imitates CircleShapeInferencePass but little bit different.
//      In CircleShapeInferencePass, DeadNodeQueryService is used to get alive nodes,
//      which are not detected with postorder_traversal.
//      However, it is not considered in this function because this is just for testing
//      inference rule itself, not for inference pass.
bool circle_shape_pass(loco::Graph *g)
{
  luci::sinf::Rule shape_infer_rule;
  bool changed = false;

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    loco::TensorShape shape;
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    if (shape_infer_rule.infer(circle_node, shape) && !is_same_shape(circle_node, shape))
    {
      circle_node->rank(shape.rank());
      for (uint32_t i = 0; i < shape.rank(); ++i)
        circle_node->dim(i) = shape.dim(i);
      circle_node->shape_status(luci::ShapeStatus::VALID);
      changed = true;
    }
  }

  return changed;
}

} // namespace

TEST(CircleShapeInferenceRuleTest, minimal_with_CircleRelu)
{
  // Create a simple network
  luci::test::TestGraph graph;
  auto relu_node = graph.append<luci::CircleRelu>(graph.input_node);
  graph.complete(relu_node);

  // set shape
  {
    graph.input_node->rank(2);
    graph.input_node->dim(0) = 3;
    graph.input_node->dim(1) = 4;

    graph.output_node->rank(2);
    graph.output_node->dim(0) = 3;
    graph.output_node->dim(1) = 4;

    luci::test::graph_input_shape(graph.input_node);
    luci::test::graph_output_shape(graph.output_node);
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(relu_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(relu_node));
    ASSERT_EQ(loco::Domain::Tensor, luci::shape_get(relu_node).domain());

    auto shape = luci::shape_get(relu_node).as<loco::TensorShape>();
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
  auto avg_node = graph.append<luci::CircleAveragePool2D>(graph.input_node);
  graph.complete();

  auto input_node = graph.input_node;
  {
    input_node->shape({1, 4, 3, 1});
    luci::test::graph_input_shape(input_node);
  }
  auto output_node = graph.output_node;
  {
    output_node->shape({1, 2, 1, 1});
    luci::test::graph_output_shape(output_node);
  }
  // setting CircleAveragePool2D
  {
    avg_node->filter()->h(2);
    avg_node->filter()->w(2);
    avg_node->stride()->h(2);
    avg_node->stride()->w(2);
    avg_node->fusedActivationFunction(luci::FusedActFunc::NONE);
    avg_node->padding(luci::Padding::VALID);
  }
  ASSERT_FALSE(loco::shape_known(avg_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(avg_node));
    ASSERT_EQ(loco::Domain::Tensor, luci::shape_get(avg_node).domain());

    auto shape = luci::shape_get(avg_node).as<loco::TensorShape>();
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
  auto avg_node = graph.append<luci::CircleAveragePool2D>(graph.input_node);
  graph.complete();

  auto input_node = graph.input_node;
  {
    input_node->shape({1, 4, 3, 1});
    luci::test::graph_input_shape(input_node);
  }
  auto output_node = graph.output_node;
  {
    output_node->shape({1, 2, 2, 1});
    luci::test::graph_output_shape(output_node);
  }

  // setting CircleAveragePool2D
  {
    avg_node->filter()->h(2);
    avg_node->filter()->w(2);
    avg_node->stride()->h(2);
    avg_node->stride()->w(2);
    avg_node->fusedActivationFunction(luci::FusedActFunc::NONE);
    avg_node->padding(luci::Padding::SAME);
  }

  ASSERT_FALSE(loco::shape_known(avg_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(avg_node));
    ASSERT_EQ(loco::Domain::Tensor, luci::shape_get(avg_node).domain());

    auto shape = luci::shape_get(avg_node).as<loco::TensorShape>();
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

  auto x_node = g->nodes()->create<luci::CircleInput>();
  {
    x_node->rank(3);
    x_node->dim(0) = 2;
    x_node->dim(1) = 1;
    x_node->dim(2) = 5;
  }
  auto y_node = g->nodes()->create<luci::CircleInput>();
  {
    y_node->rank(2);
    y_node->dim(0) = 3;
    y_node->dim(1) = 5;
  }
  auto add_node = g->nodes()->create<luci::CircleAdd>();
  {
    add_node->x(x_node);
    add_node->y(y_node);
  }
  auto output_node = g->nodes()->create<luci::CircleOutput>();
  {
    output_node->from(add_node);
  }

  auto x_input = g->inputs()->create();
  {
    x_input->name("x");
    luci::link(x_input, x_node);
  }
  auto y_input = g->inputs()->create();
  {
    y_input->name("y");
    luci::link(y_input, y_node);
  }
  auto output = g->outputs()->create();
  {
    output->name("output");
    luci::link(output, output_node);
  }

  luci::test::graph_input_shape(x_node);
  luci::test::graph_input_shape(y_node);
  luci::test::graph_output_shape(output_node);

  // pre-check
  ASSERT_FALSE(loco::shape_known(add_node));

  // shape inference
  while (shape_pass(g.get()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(add_node));
    ASSERT_EQ(loco::Domain::Tensor, luci::shape_get(add_node).domain());

    auto shape = luci::shape_get(add_node).as<loco::TensorShape>();
    ASSERT_EQ(3, shape.rank());
    ASSERT_EQ(2, shape.dim(0));
    ASSERT_EQ(3, shape.dim(1));
    ASSERT_EQ(5, shape.dim(2));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleTranspose_simple)
{
  luci::test::ExampleGraph<luci::test::ExampleGraphType::CircleTranspose> g;

  g.input_node->rank(3);
  g.input_node->dim(0) = 3;
  g.input_node->dim(1) = 8;
  g.input_node->dim(2) = 1;

  g.const_perm->dtype(loco::DataType::S32);
  g.const_perm->rank(1);
  g.const_perm->dim(0) = 3;
  g.const_perm->size<loco::DataType::S32>(3);
  g.const_perm->at<loco::DataType::S32>(0) = 1;
  g.const_perm->at<loco::DataType::S32>(1) = 2;
  g.const_perm->at<loco::DataType::S32>(2) = 0;

  luci::test::graph_input_shape(g.input_node);
  luci::test::graph_output_shape(g.output_node);

  // pre-check
  ASSERT_FALSE(loco::shape_known(g.transpose_node));

  // shape inference
  while (shape_pass(g.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(g.transpose_node));

    auto shape = luci::shape_get(g.transpose_node).as<loco::TensorShape>();
    ASSERT_EQ(3, shape.rank());
    ASSERT_EQ(8, shape.dim(0));
    ASSERT_EQ(1, shape.dim(1));
    ASSERT_EQ(3, shape.dim(2));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleSqueeze)
{
  luci::test::TestGraph graph;
  auto squeeze_node = graph.append<luci::CircleSqueeze>(graph.input_node);
  graph.complete();

  auto input_node = graph.input_node;
  {
    input_node->shape({1, 4, 3, 1});
  }
  auto output_node = graph.output_node;
  {
    output_node->shape({4, 3, 1});
  }

  luci::test::graph_input_shape(input_node);
  luci::test::graph_output_shape(output_node);

  squeeze_node->squeeze_dims({0});

  // pre-check
  ASSERT_FALSE(loco::shape_known(squeeze_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(squeeze_node));

    auto shape = luci::shape_get(squeeze_node).as<loco::TensorShape>();
    ASSERT_EQ(3, shape.rank());
    ASSERT_EQ(4, shape.dim(0));
    ASSERT_EQ(3, shape.dim(1));
    ASSERT_EQ(1, shape.dim(2));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleExpandDims)
{
  luci::test::TestGraph graph;
  auto axis = graph.append<luci::CircleConst>();
  axis->dtype(loco::DataType::S32);
  axis->rank(0);
  axis->size<loco::DataType::S32>(1);
  axis->at<loco::DataType::S32>(0) = 1;

  auto expand_dims = graph.append<luci::CircleExpandDims>(graph.input_node, axis);
  graph.complete();

  auto input_node = graph.input_node;
  {
    input_node->shape({4, 3});
  }

  auto output_node = graph.output_node;
  {
    output_node->from(expand_dims);
  }

  luci::test::graph_input_shape(input_node);
  luci::test::graph_output_shape(output_node);

  // shape inference
  while (shape_pass(graph.graph()))
    ;

  // validation
  {
    ASSERT_TRUE(loco::shape_known(expand_dims));

    auto shape = luci::shape_get(expand_dims).as<loco::TensorShape>();

    ASSERT_EQ(3, shape.rank());
    ASSERT_EQ(4, shape.dim(0));
    ASSERT_EQ(1, shape.dim(1));
    ASSERT_EQ(3, shape.dim(2));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleSqueezeAll)
{
  luci::test::TestGraph graph;
  auto squeeze_node = graph.append<luci::CircleSqueeze>(graph.input_node);
  graph.complete();

  auto input_node = graph.input_node;
  {
    input_node->shape({1, 4, 3, 1});
  }
  auto output_node = graph.output_node;
  {
    input_node->shape({4, 3});
  }

  luci::test::graph_input_shape(input_node);
  luci::test::graph_output_shape(output_node);

  squeeze_node->squeeze_dims({});

  // pre-check
  ASSERT_FALSE(loco::shape_known(squeeze_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(squeeze_node));

    auto shape = luci::shape_get(squeeze_node).as<loco::TensorShape>();
    ASSERT_EQ(2, shape.rank());
    ASSERT_EQ(4, shape.dim(0));
    ASSERT_EQ(3, shape.dim(1));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleGatherNd_simple)
{
  luci::test::TestGraph graph;
  auto indices_const = graph.append<luci::CircleConst>();
  auto gather_nd_node = graph.append<luci::CircleGatherNd>(graph.input_node, indices_const);
  graph.complete();

  {
    auto input_node = graph.input_node;
    input_node->shape({1, 4, 4, 3});
    luci::test::graph_input_shape(input_node);
  }
  {
    auto output_node = graph.output_node;
    output_node->shape({1, 2, 2, 3});
    luci::test::graph_output_shape(output_node);
  }

  {
    indices_const->shape({1, 2, 3});
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(gather_nd_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(gather_nd_node));

    auto shape = luci::shape_get(gather_nd_node).as<loco::TensorShape>();
    ASSERT_EQ(3, shape.rank());
    ASSERT_EQ(1, shape.dim(0));
    ASSERT_EQ(2, shape.dim(1));
    ASSERT_EQ(3, shape.dim(2));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleGatherNd_slices)
{
  luci::test::TestGraph graph;
  auto indices_const = graph.append<luci::CircleConst>();
  auto gather_nd_node = graph.append<luci::CircleGatherNd>(graph.input_node, indices_const);
  graph.complete();

  {
    auto input_node = graph.input_node;
    input_node->shape({1, 4, 4, 3});
    luci::test::graph_input_shape(input_node);
  }
  {
    auto output_node = graph.output_node;
    output_node->shape({1, 2, 4, 4, 3});
    luci::test::graph_output_shape(output_node);
  }

  {
    indices_const->shape({1, 2, 1});
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(gather_nd_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(gather_nd_node));

    auto shape = luci::shape_get(gather_nd_node).as<loco::TensorShape>();
    ASSERT_EQ(5, shape.rank());
    ASSERT_EQ(1, shape.dim(0));
    ASSERT_EQ(2, shape.dim(1));
    ASSERT_EQ(4, shape.dim(2));
    ASSERT_EQ(4, shape.dim(3));
    ASSERT_EQ(3, shape.dim(4));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleGatherNd_NEG)
{
  luci::test::TestGraph graph;
  auto indices_const = graph.append<luci::CircleConst>();
  auto gather_nd_node = graph.append<luci::CircleGatherNd>(graph.input_node, indices_const);
  graph.complete();

  {
    auto input_node = graph.input_node;
    input_node->shape({1, 4, 4, 3});
    luci::test::graph_input_shape(input_node);
  }
  {
    // Does not matter, because test should fail anyway
    auto output_node = graph.output_node;
    output_node->shape({0, 0, 0});
    luci::test::graph_output_shape(output_node);
  }

  {
    indices_const->shape({1, 2, 5});
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(gather_nd_node));

  // had to pack into lambda to check throw
  auto lambda = [&]() {
    // shape inference
    while (shape_pass(graph.graph()) == true)
      ;
  };

  ASSERT_THROW(lambda(), oops::InternalExn);
}

TEST(CircleShapeInferenceRuleTest, CircleResizeNearestNeighbor)
{
  luci::test::TestGraph graph;
  auto size_const = graph.append<luci::CircleConst>();
  size_const->dtype(loco::DataType::S32);
  size_const->rank(1);
  size_const->dim(0) = 2;
  size_const->size<loco::DataType::S32>(2);
  size_const->at<loco::DataType::S32>(0) = 16;
  size_const->at<loco::DataType::S32>(1) = 16;
  auto resize_node = graph.append<luci::CircleResizeNearestNeighbor>(graph.input_node, size_const);
  graph.complete();

  {
    auto input_node = graph.input_node;
    input_node->shape({1, 4, 4, 3});
    luci::test::graph_input_shape(input_node);
  }
  {
    auto output_node = graph.output_node;
    output_node->from(resize_node);
    luci::test::graph_output_shape(output_node);
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(resize_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(resize_node));

    auto shape = luci::shape_get(resize_node).as<loco::TensorShape>();
    ASSERT_EQ(4, shape.rank());
    ASSERT_EQ(1, shape.dim(0));
    ASSERT_EQ(16, shape.dim(1));
    ASSERT_EQ(16, shape.dim(2));
    ASSERT_EQ(3, shape.dim(3));
  }
}

TEST(CircleShapeInferenceRuleTest, CircleResizeBilinear)
{
  luci::test::TestGraph graph;
  auto size_const = graph.append<luci::CircleConst>();
  size_const->dtype(loco::DataType::S32);
  size_const->rank(1);
  size_const->dim(0) = 2;
  size_const->size<loco::DataType::S32>(2);
  size_const->at<loco::DataType::S32>(0) = 16;
  size_const->at<loco::DataType::S32>(1) = 16;
  auto resize_node = graph.append<luci::CircleResizeBilinear>(graph.input_node, size_const);
  graph.complete();

  {
    auto input_node = graph.input_node;
    input_node->shape({1, 4, 4, 3});
    luci::test::graph_input_shape(input_node);
  }
  {
    auto output_node = graph.output_node;
    output_node->from(resize_node);
    luci::test::graph_output_shape(output_node);
  }

  // pre-check
  ASSERT_FALSE(loco::shape_known(resize_node));

  // shape inference
  while (shape_pass(graph.graph()) == true)
    ;

  // Verify
  {
    ASSERT_TRUE(loco::shape_known(resize_node));

    auto shape = luci::shape_get(resize_node).as<loco::TensorShape>();
    ASSERT_EQ(4, shape.rank());
    ASSERT_EQ(1, shape.dim(0));
    ASSERT_EQ(16, shape.dim(1));
    ASSERT_EQ(16, shape.dim(2));
    ASSERT_EQ(3, shape.dim(3));
  }
}
