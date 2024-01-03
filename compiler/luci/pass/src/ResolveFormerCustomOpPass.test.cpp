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
class CustomBroadcastToGraphlet
{
public:
  CustomBroadcastToGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    // CircleCustom(BroadcastTo)
    _broadcastTo = g->nodes()->create<luci::CircleCustom>(2, 1);
    _broadcastTo->custom_code("BroadcastTo");
    _broadcastTo->dtype(loco::DataType::FLOAT32);
    _broadcastTo->shape({2, 3});
    _broadcastTo->name("BroadcastTo");

    // CircleConst(BroadcastTo-input)
    _input = g->nodes()->create<luci::CircleConst>();
    _input->dtype(loco::DataType::FLOAT32);
    _input->shape({1, 3});
    _input->size<loco::DataType::FLOAT32>(3);
    _input->at<loco::DataType::FLOAT32>(0) = 1;
    _input->at<loco::DataType::FLOAT32>(1) = 2;
    _input->at<loco::DataType::FLOAT32>(2) = 3;

    // CircleConst(BroadcastTo-shape)
    _shape = g->nodes()->create<luci::CircleConst>();
    _shape->dtype(loco::DataType::S32);
    _shape->shape({2});
    _shape->size<loco::DataType::S32>(2);
    _shape->at<loco::DataType::S32>(0) = 2;
    _shape->at<loco::DataType::S32>(1) = 3;

    _broadcastTo->inputs(0, _input);
    _broadcastTo->inputs(1, _shape);

    // CircleCustomOut
    _broadcastTo_out = g->nodes()->create<luci::CircleCustomOut>();
    _broadcastTo_out->shape({2, 3});
    _broadcastTo_out->dtype(loco::DataType::FLOAT32);
    _broadcastTo_out->index(0);
    _broadcastTo_out->input(_broadcastTo);
  }

public:
  luci::CircleCustom *broadcastTo() { return _broadcastTo; }
  luci::CircleConst *shape() { return _shape; }
  luci::CircleCustomOut *broadcastTo_out() { return _broadcastTo_out; }

protected:
  luci::CircleCustom *_broadcastTo = nullptr;
  luci::CircleCustomOut *_broadcastTo_out = nullptr;
  luci::CircleConst *_input = nullptr;
  luci::CircleConst *_shape = nullptr;
};

class BroadcastToGraph : public TestIGraphlet,
                         public TestOsGraphlet<1>,
                         public CustomBroadcastToGraphlet
{
public:
  BroadcastToGraph() = default;

  void init(void)
  {
    TestOsGraphlet<1>::init(g(), {{1, 2, 3, 1, 2, 3}});
    CustomBroadcastToGraphlet::init(g());

    output(0)->from(_broadcastTo_out);
  }
};

class FormerCustomOpGraphTest : public ::testing::Test
{
public:
  BroadcastToGraph _g;
  luci::ResolveFormerCustomOpPass _pass;
};

} // namespace

TEST(ResolveFormerCustomOpPassTest, name)
{
  luci::ResolveFormerCustomOpPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FormerCustomOpGraphTest, simple_test_BroadcastTo)
{
  _g.init();

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(true, ret);

  auto broadcastTo = dynamic_cast<luci::CircleBroadcastTo *>(_g.output(0)->from());
  EXPECT_NE(nullptr, broadcastTo);

  auto input = dynamic_cast<luci::CircleConst *>(broadcastTo->input());
  EXPECT_NE(nullptr, input);
  EXPECT_EQ(1, input->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(2, input->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(3, input->at<loco::DataType::FLOAT32>(2));

  auto shape = dynamic_cast<luci::CircleConst *>(broadcastTo->shape());
  EXPECT_NE(nullptr, shape);
  EXPECT_EQ(true, (shape->dtype() == loco::DataType::S32));
  EXPECT_EQ(2, shape->at<loco::DataType::S32>(0));
  EXPECT_EQ(3, shape->at<loco::DataType::S32>(1));
}

TEST_F(FormerCustomOpGraphTest, wrong_op_NEG)
{
  _g.init();

  _g.broadcastTo()->custom_code("UNSUPORTED_CUSTOM_CODE");

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}

TEST_F(FormerCustomOpGraphTest, wrong_shape_type_NEG)
{
  // the data type of shape should be S32 or S64.
  _g.init();

  _g.shape()->dtype(loco::DataType::FLOAT32);

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}

TEST_F(FormerCustomOpGraphTest, unequal_input_output_type_NEG)
{
  _g.init();

  _g.broadcastTo_out()->dtype(loco::DataType::S32);

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}
