/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ResolveBuiltinOpAddPass.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/CircleNodes.h>
#include <gtest/gtest.h>

using namespace luci::test;

namespace
{

/**
 *  graph having Built-in operator BroadcastTo
 *
 *
 *        [Input]   [BroadcastTo]
 *             \         /
 *             [CircleAdd]
 *                  |
 *             [CustomOut]
 *                  |
 *              [Output]
 */
class BuiltinOpAddGraphlet
{
public:
  BuiltinOpAddGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    // CircleCustom(AddV2)
    _addV2 = g->nodes()->create<luci::CircleCustom>(2, 1);
    _addV2->custom_code("AddV2");
    _addV2->shape({2, 3});
    _addV2->dtype(loco::DataType::FLOAT32);
    _addV2->name("addV2");

    // BroadcastTo
    _broadcastTo = g->nodes()->create<luci::CircleBroadcastTo>();
    _broadcastTo->dtype(loco::DataType::FLOAT32);
    _broadcastTo->name("BroadcastTo");

    // CircleConst (BroadcastTo-input)
    auto input = g->nodes()->create<luci::CircleConst>();
    input->dtype(loco::DataType::FLOAT32);
    input->shape({1, 3});
    input->size<loco::DataType::FLOAT32>(3);
    input->at<loco::DataType::FLOAT32>(0) = 1;
    input->at<loco::DataType::FLOAT32>(1) = 2;
    input->at<loco::DataType::FLOAT32>(2) = 3;

    // CircleConst (BroadcastTo-shape)
    auto shape = g->nodes()->create<luci::CircleConst>();
    shape->dtype(loco::DataType::S32);
    shape->shape({2});
    shape->size<loco::DataType::S32>(2);
    shape->at<loco::DataType::S32>(0) = 2;
    shape->at<loco::DataType::S32>(1) = 3;

    _broadcastTo->input(input);
    _broadcastTo->shape(shape);

    _addV2->inputs(1, _broadcastTo);

    // CircleCustomOut
    _addV2_out = g->nodes()->create<luci::CircleCustomOut>();
    _addV2_out->shape({2, 3});
    _addV2_out->dtype(loco::DataType::FLOAT32);
    _addV2_out->index(0);
    _addV2_out->input(_addV2);
  }

public:
  luci::CircleCustom *addV2() { return _addV2; }
  luci::CircleBroadcastTo *broadcastTo() { return _broadcastTo; }

protected:
  luci::CircleCustom *_addV2 = nullptr;
  luci::CircleCustomOut *_addV2_out = nullptr;
  luci::CircleBroadcastTo *_broadcastTo = nullptr;
};

class BuiltinOpAddV2Graph : public TestIGraphlet,
                            public TestOsGraphlet<1>,
                            public BuiltinOpAddGraphlet
{
public:
  BuiltinOpAddV2Graph() = default;

  void init(void)
  {
    TestIGraphlet::init(g(), {2, 3});
    TestOsGraphlet<1>::init(g(), {{2, 3}});
    BuiltinOpAddGraphlet::init(g());

    // connect graph
    _addV2->inputs(0, input());

    output(0)->from(_addV2_out);
  }
};

class BuitlinAddV2GraphTest : public ::testing::Test
{
public:
  BuiltinOpAddV2Graph _g;
  luci::ResolveBuiltinOpAddPass _pass;
};

} // namespace

TEST_F(BuitlinAddV2GraphTest, name_test)
{
  auto const name = _pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(BuitlinAddV2GraphTest, simple_test_BroadcastToAddV2)
{
  _g.init();

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(true, ret);

  auto add = dynamic_cast<luci::CircleAdd *>(_g.output(0)->from());
  EXPECT_NE(nullptr, add);

  auto broadcastTo_input = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, broadcastTo_input);
  EXPECT_EQ(1, broadcastTo_input->at<loco::DataType::FLOAT32>(0));
  EXPECT_EQ(2, broadcastTo_input->at<loco::DataType::FLOAT32>(1));
  EXPECT_EQ(3, broadcastTo_input->at<loco::DataType::FLOAT32>(2));

  auto input = dynamic_cast<luci::CircleNode *>(add->x());
  EXPECT_NE(nullptr, input);
  EXPECT_EQ(2, input->dim(0));
  EXPECT_EQ(3, input->dim(1));
}

TEST_F(BuitlinAddV2GraphTest, wrong_custom_code_NEG)
{
  _g.init();
  _g.addV2()->custom_code("UNSUPORTED_CUSTOM_CODE");

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}

TEST_F(BuitlinAddV2GraphTest, wrong_input_NEG)
{
  _g.init();

  _g.addV2()->inputs(0, nullptr);

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}

TEST_F(BuitlinAddV2GraphTest, wrong_input_type_NEG)
{
  _g.init();

  _g.broadcastTo()->dtype(loco::DataType::BOOL);

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}
