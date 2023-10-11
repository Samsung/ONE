/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ResolveFormerCustomOpPass.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/CircleNodes.h>
#include <gtest/gtest.h>

using namespace luci::test;

namespace
{

/**
 *  graph having Custom operator BroadcastTo
 *
 *     [Const(Input)] [Const(Shape)]
 *             \         /
 *         [Custom(BroadcastTo)]
 *                  |
 *             [CustomOut]
 *                  |
 *              [Output]
 */
class BroadcastToGraphlet
{
public:
  BroadcastToGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    // CircleCustom(BroadcastTo)
    _broadcastTo = g->nodes()->create<luci::CircleCustom>(2, 1);
    _broadcastTo->custom_code("BroadcastTo");
    _broadcastTo->dtype(loco::DataType::FLOAT32);
    _broadcastTo->shape({2, 3});
    _broadcastTo->name("BroadcastTo");

    // CircleConst
    auto input = g->nodes()->create<luci::CircleConst>();
    input->dtype(loco::DataType::FLOAT32);
    input->shape({1, 3});
    input->size<loco::DataType::FLOAT32>(3);
    input->at<loco::DataType::FLOAT32>(0) = 1;
    input->at<loco::DataType::FLOAT32>(1) = 2;
    input->at<loco::DataType::FLOAT32>(2) = 3;

    // CircleConst
    auto shape = g->nodes()->create<luci::CircleConst>();
    shape->dtype(loco::DataType::S32);
    shape->shape({2});
    shape->size<loco::DataType::S32>(2);
    shape->at<loco::DataType::S32>(0) = 2;
    shape->at<loco::DataType::S32>(1) = 3;

    _broadcastTo->inputs(0, input);
    _broadcastTo->inputs(1, shape);

    // CircleCustomOut
    _broadcastTo_out = g->nodes()->create<luci::CircleCustomOut>();
    _broadcastTo_out->shape({2, 3});
    _broadcastTo_out->dtype(loco::DataType::FLOAT32);
    _broadcastTo_out->index(0);
    _broadcastTo_out->input(_broadcastTo);
  }

public:
  luci::CircleCustom *broadcastTo() { return _broadcastTo; }

protected:
  luci::CircleCustom *_broadcastTo = nullptr;
  luci::CircleCustomOut *_broadcastTo_out = nullptr;
};

class BroadcastToGraph : public TestIGraphlet, public TestOsGraphlet<1>, public BroadcastToGraphlet
{
public:
  BroadcastToGraph() = default;

  void init(void)
  {
    TestOsGraphlet<1>::init(g(), {{1, 2, 3, 1, 2, 3}});
    BroadcastToGraphlet::init(g());

    output(0)->from(_broadcastTo_out);
  }
};

class FormerCustomOpGraphTest : public ::testing::Test
{
public:
  BroadcastToGraph g;
  luci::ResolveFormerCustomOpPass pass;
};

} // namespace

TEST_F(FormerCustomOpGraphTest, name)
{
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FormerCustomOpGraphTest, simple_test_BroadcastTo)
{
  g.init();

  auto ret = pass.run(g.g());
  EXPECT_EQ(true, ret);

  auto broadcastTo = dynamic_cast<luci::CircleBroadcastTo *>(g.output(0)->from());
  EXPECT_NE(nullptr, broadcastTo);

  auto input = loco::must_cast<luci::CircleConst *>(broadcastTo->input());
  EXPECT_NE(nullptr, input);
  EXPECT_EQ(1, input->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(2, input->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(3, input->at<loco::DataType::FLOAT32>(2));

  auto shape = loco::must_cast<luci::CircleConst *>(broadcastTo->shape());
  EXPECT_NE(nullptr, shape);
  EXPECT_EQ(true, (shape->dtype() == loco::DataType::S32 || shape->dtype() == loco::DataType::S64));
  EXPECT_EQ(2, shape->at<loco::DataType::S32>(0));
  EXPECT_EQ(3, shape->at<loco::DataType::S32>(1));
}

TEST_F(FormerCustomOpGraphTest, wrong_op_NEG)
{
  g.init();

  g.broadcastTo()->custom_code("Abs");

  auto ret = pass.run(g.g());
  EXPECT_EQ(false, ret);
}
