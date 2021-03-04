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

#include "test/TestIOGraph.h"

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
