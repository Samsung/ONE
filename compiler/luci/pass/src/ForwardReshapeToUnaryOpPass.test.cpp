/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ForwardReshapeToUnaryOpPass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

#include <vector>

namespace
{

using namespace luci::test;

class ReshapeNegGraphlet
{
public:
  ReshapeNegGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    std::vector<uint32_t> shape_out_v = shape_out;

    _reshape_shape = g->nodes()->create<luci::CircleConst>();
    _reshape = g->nodes()->create<luci::CircleReshape>();
    _neg = g->nodes()->create<luci::CircleNeg>();

    _reshape_shape->dtype(loco::DataType::S32);
    _reshape_shape->rank(1);
    _reshape_shape->dim(0).set(shape_out_v.size());
    _reshape_shape->shape_status(luci::ShapeStatus::VALID);
    // values
    const auto size = shape_out_v.size();
    _reshape_shape->size<loco::DataType::S32>(size);
    for (uint32_t i = 0; i < size; i++)
      _reshape_shape->at<loco::DataType::S32>(i) = shape_out_v[i];

    _reshape_shape->name("reshape_shape");
    _reshape->name("reshape");
    _neg->name("neg");
  }

protected:
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleNeg *_neg = nullptr;
  luci::CircleConst *_reshape_shape = nullptr;
};

// TODO Reduce duplicate code with ReshapeNegGraphlet
class ReshapeLogisticGraphlet
{
public:
  ReshapeLogisticGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    std::vector<uint32_t> shape_out_v = shape_out;

    _reshape_shape = g->nodes()->create<luci::CircleConst>();
    _reshape = g->nodes()->create<luci::CircleReshape>();
    _logistic = g->nodes()->create<luci::CircleLogistic>();

    _reshape_shape->dtype(loco::DataType::S32);
    _reshape_shape->rank(1);
    _reshape_shape->dim(0).set(shape_out_v.size());
    _reshape_shape->shape_status(luci::ShapeStatus::VALID);
    // values
    const auto size = shape_out_v.size();
    _reshape_shape->size<loco::DataType::S32>(size);
    for (uint32_t i = 0; i < size; i++)
      _reshape_shape->at<loco::DataType::S32>(i) = shape_out_v[i];

    _reshape_shape->name("reshape_shape");
    _reshape->name("reshape");
    _logistic->name("logistic");
  }

protected:
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleLogistic *_logistic = nullptr;
  luci::CircleConst *_reshape_shape = nullptr;
};

class ForwardReshapeToNegGraph : public TestIOGraph, public ReshapeNegGraphlet
{
public:
  ForwardReshapeToNegGraph() = default;

public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    ReshapeNegGraphlet::init(g(), shape_in, shape_out);

    // connect network
    _reshape->tensor(input());
    _reshape->shape(_reshape_shape);
    _neg->x(_reshape);

    output()->from(_neg);
  }
};

class ForwardReshapeToLogisticGraph : public TestIOGraph, public ReshapeLogisticGraphlet
{
public:
  ForwardReshapeToLogisticGraph() = default;

public:
  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    ReshapeLogisticGraphlet::init(g(), shape_in, shape_out);

    // connect network
    _reshape->tensor(input());
    _reshape->shape(_reshape_shape);
    _logistic->x(_reshape);

    output()->from(_logistic);
  }
};

class ForwardReshapeToNegGraphTest : public ::testing::Test
{
public:
  ForwardReshapeToNegGraphTest() = default;

  void run_pass(void)
  {
    while (_pass.run(_graph.g()))
      ;
  }

protected:
  ForwardReshapeToNegGraph _graph;
  luci::ForwardReshapeToUnaryOpPass _pass;
};

class ForwardReshapeToLogisticGraphTest : public ::testing::Test
{
public:
  ForwardReshapeToLogisticGraphTest() = default;

  void run_pass(void)
  {
    while (_pass.run(_graph.g()))
      ;
  }

protected:
  ForwardReshapeToLogisticGraph _graph;
  luci::ForwardReshapeToUnaryOpPass _pass;
};

/**
 *  Simple graph for test
 *
 *  BEFORE
 *               [Input]
 *              (3, 4, 4)                 [Shape_Const = (1, -1, 4)]
 *                  |                     |
 *              [Reshape] ----------------
 *              (1, 12, 4)
 *                  |
 *        [Mean, keep_dims = true]
 *              (1, 12, 1)
 *                  |
 *               [Output]
 *
 *  AFTER
 *               [Input]
 *              (3, 4, 4)
 *                  |
 *         [Mean, keep_dims = true]
 *              (3, 4, 1)                 [Shape_Const = (1, -1, 1)]
 *                  |                     |
 *              [Reshape]-----------------
 *              (1, 12, 1)
 *                  |
 *              [Output]
 *
 */
class PatternReshapeMeanGraphlet
{
public:
  PatternReshapeMeanGraphlet() = default;

  void init(loco::Graph *g)
  {
    _mean = g->nodes()->create<luci::CircleMean>();
    _mean_const = g->nodes()->create<luci::CircleConst>();
    _reshape = g->nodes()->create<luci::CircleReshape>();
    _reshape_const = g->nodes()->create<luci::CircleConst>();

    _mean->name("_mean");
    _mean_const->name("_mean_const");
    _reshape->name("_reshape");
    _reshape_const->name("_reshape_const");
  }

public:
  luci::CircleMean *mean() { return _mean; }
  luci::CircleConst *mean_const() { return _mean_const; }
  luci::CircleReshape *reshape() { return _reshape; }
  luci::CircleConst *reshape_const() { return _reshape_const; }

protected:
  luci::CircleMean *_mean = nullptr;
  luci::CircleConst *_mean_const = nullptr;
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleConst *_reshape_const = nullptr;
};

class ForwardReshapeToMeanPatternTestGraph : public TestIOGraph, public PatternReshapeMeanGraphlet
{
public:
  ForwardReshapeToMeanPatternTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({3, 4, 4}, {3, 4, 4});
    PatternReshapeMeanGraphlet::init(g());

    _reshape_const->rank(1);
    _reshape_const->dtype(loco::DataType::S32);
    _reshape_const->size<loco::DataType::S32>(3);
    _reshape_const->at<loco::DataType::S32>(0) = 1;
    _reshape_const->at<loco::DataType::S32>(1) = -1;
    _reshape_const->at<loco::DataType::S32>(2) = 4;
    _reshape_const->shape_status(luci::ShapeStatus::VALID);

    _reshape->rank(3);
    _reshape->dim(0).set(3);
    _reshape->dim(1).set(4);
    _reshape->dim(2).set(4);
    _reshape->dtype(loco::DataType::FLOAT32);
    _reshape->shape_status(luci::ShapeStatus::VALID);
    _reshape->tensor(input());
    _reshape->shape(_reshape_const);

    _mean_const->rank(1);
    _mean_const->dtype(loco::DataType::S32);
    _mean_const->size<loco::DataType::S32>(1);
    _mean_const->at<loco::DataType::S32>(0) = -1;
    _mean_const->shape_status(luci::ShapeStatus::VALID);

    _mean->rank(3);
    _mean->dim(0).set(1);
    _mean->dim(1).set(12);
    _mean->dim(2).set(1);
    _mean->dtype(loco::DataType::FLOAT32);
    _mean->shape_status(luci::ShapeStatus::VALID);
    _mean->input(_reshape);
    _mean->reduction_indices(_mean_const);
    _mean->keep_dims(true);

    output()->from(_mean);
  }

  void invalid_type() { _mean_const->dtype(loco::DataType::FLOAT32); }
};

} // namespace

TEST(ForwardReshapeToUnaryOpPassTest, name)
{
  luci::ForwardReshapeToUnaryOpPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(ForwardReshapeToNegGraphTest, simple_forward)
{
  _graph.init({2, 2, 2}, {2, 4});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto neg = dynamic_cast<luci::CircleNeg *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, neg);
  neg = dynamic_cast<luci::CircleNeg *>(reshape->tensor());
  ASSERT_NE(nullptr, neg);
}

TEST_F(ForwardReshapeToLogisticGraphTest, forward)
{
  _graph.init({2, 2, 2}, {2, 4});

  run_pass();

  auto reshape = dynamic_cast<luci::CircleReshape *>(_graph.output()->from());
  auto log = dynamic_cast<luci::CircleLogistic *>(_graph.output()->from());
  ASSERT_NE(nullptr, reshape);
  ASSERT_EQ(nullptr, log);
  log = dynamic_cast<luci::CircleLogistic *>(reshape->tensor());
  ASSERT_NE(nullptr, log);
}

TEST(FuseMulWithDivPassTest, forward_reshape_to_mean_pattern)
{
  ForwardReshapeToMeanPatternTestGraph g;
  luci::ForwardReshapeToUnaryOpPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseMulWithDivPassTest, forward_reshape_to_mean_pattern_NEG)
{
  ForwardReshapeToMeanPatternTestGraph g;
  luci::ForwardReshapeToUnaryOpPass pass;

  g.init();

  g.invalid_type();

  EXPECT_FALSE(pass.run(g.g()));
}
