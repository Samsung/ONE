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

#include "luci/Pass/ResolveCustomOpAddPass.h"

#include <gtest/gtest.h>
#include <luci/test/TestIOGraph.h>

#include <luci/IR/CircleNodes.h>

using namespace luci::test;

namespace
{

/**
 *   Test graph with Custom(AddV2) to resolve
 *
 * [Pattern 1]
 *        [Input]   [BroadcastTo]
 *             \         /
 *           [Custom(AddV2)]
 *                  |
 *             [CustomOut]
 *                  |
 *               [Output]
 *
 * [Pattern 2]
 *        [Input]   [Custom(BroadcastTo)]
 *             \         /
 *           [Custom(AddV2)]
 *                  |
 *             [CustomOut]
 *                  |
 *               [Output]
 */
class BroadcastToAddGraphlet
{
public:
  BroadcastToAddGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    _addV2 = g->nodes()->create<luci::CircleCustom>(2, 1);
    _addV2->custom_code("AddV2");
    _addV2->shape({2, 3});
    _addV2->dtype(loco::DataType::FLOAT32);
    _addV2->name("addV2");

    // Const as BroadcastTo input
    _broadcastTo_input = g->nodes()->create<luci::CircleConst>();
    _broadcastTo_input->dtype(loco::DataType::FLOAT32);
    _broadcastTo_input->shape({1, 3});
    _broadcastTo_input->size<loco::DataType::FLOAT32>(3);
    _broadcastTo_input->at<loco::DataType::FLOAT32>(0) = 1;
    _broadcastTo_input->at<loco::DataType::FLOAT32>(1) = 2;
    _broadcastTo_input->at<loco::DataType::FLOAT32>(2) = 3;

    // Const as BroadcastTo shape
    auto broadcastTo_shape = g->nodes()->create<luci::CircleConst>();
    broadcastTo_shape->dtype(loco::DataType::S32);
    broadcastTo_shape->shape({2});
    broadcastTo_shape->size<loco::DataType::S32>(2);
    broadcastTo_shape->at<loco::DataType::S32>(0) = 2;
    broadcastTo_shape->at<loco::DataType::S32>(1) = 3;

    _custom_broadcastTo = g->nodes()->create<luci::CircleCustom>(2, 1);
    _custom_broadcastTo->custom_code("BroadcastTo");
    _custom_broadcastTo->dtype(loco::DataType::FLOAT32);
    _custom_broadcastTo->shape({2, 3});
    _custom_broadcastTo->name("BroadcastTo");

    _custom_broadcastTo->inputs(0, _broadcastTo_input);
    _custom_broadcastTo->inputs(1, broadcastTo_shape);

    _custom_broadcastTo_out = g->nodes()->create<luci::CircleCustomOut>();
    _custom_broadcastTo->custom_code("BroadcastTo");
    _custom_broadcastTo_out->shape({2, 3});
    _custom_broadcastTo_out->dtype(loco::DataType::FLOAT32);
    _custom_broadcastTo_out->index(0);
    _custom_broadcastTo_out->input(_custom_broadcastTo);

    _builtin_broadcastTo = g->nodes()->create<luci::CircleBroadcastTo>();
    _builtin_broadcastTo->dtype(loco::DataType::FLOAT32);
    _builtin_broadcastTo->name("BroadcastTo");

    _builtin_broadcastTo->input(_broadcastTo_input);
    _builtin_broadcastTo->shape(broadcastTo_shape);

    _addV2_out = g->nodes()->create<luci::CircleCustomOut>();
    _addV2_out->shape({2, 3});
    _addV2_out->dtype(loco::DataType::FLOAT32);
    _addV2_out->index(0);
    _addV2_out->input(_addV2);
  }

public:
  luci::CircleCustom *addV2() { return _addV2; }
  luci::CircleBroadcastTo *builtin_broadcastTo() { return _builtin_broadcastTo; }

protected:
  luci::CircleCustom *_addV2 = nullptr;
  luci::CircleCustomOut *_addV2_out = nullptr;
  luci::CircleCustom *_custom_broadcastTo = nullptr;
  luci::CircleBroadcastTo *_builtin_broadcastTo = nullptr;
  luci::CircleCustomOut *_custom_broadcastTo_out = nullptr;
  luci::CircleConst *_broadcastTo_input = nullptr;
};

class BroadcastToAddV2Graph : public TestIGraphlet,
                              public TestOsGraphlet<1>,
                              public BroadcastToAddGraphlet
{
public:
  BroadcastToAddV2Graph() = default;

  void init(const bool &isCustomBroadcastTo)
  {
    TestIGraphlet::init(g(), {2, 3});
    TestOsGraphlet<1>::init(g(), {{2, 3}});
    BroadcastToAddGraphlet::init(g());

    // connect Input and Output to test graph
    _addV2->inputs(0, input());

    if (isCustomBroadcastTo)
      _addV2->inputs(1, _custom_broadcastTo_out);
    else
      _addV2->inputs(1, _builtin_broadcastTo);

    _addV2_out->input(_addV2);
    output(0)->from(_addV2_out);
  }
};

class ResolveCustomOpAddPassTest : public ::testing::Test
{
public:
  BroadcastToAddV2Graph _g;
  luci::ResolveCustomOpAddPass _pass;
};

} // namespace

TEST_F(ResolveCustomOpAddPassTest, name)
{
  auto const name = _pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(ResolveCustomOpAddPassTest, simple_test_CustomBroadcastTo)
{
  _g.init(true);

  // check if Custom(AddV2) exists before the pass
  auto addV2_out = dynamic_cast<luci::CircleCustomOut *>(_g.output(0)->from());
  EXPECT_NE(nullptr, addV2_out);
  auto addV2 = dynamic_cast<luci::CircleCustom *>(addV2_out->input());
  EXPECT_NE(nullptr, addV2);
  EXPECT_TRUE("AddV2" == addV2->custom_code());

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(true, ret);

  // check if Custom(AddV2) is converted to Add
  auto add = dynamic_cast<luci::CircleAdd *>(_g.output(0)->from());
  EXPECT_NE(nullptr, add);

  // check if Custom(BroadcastTo) is removed
  auto input_y = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, input_y);
}

TEST_F(ResolveCustomOpAddPassTest, simple_test_BuiltinBroadcastTo)
{
  _g.init(false);

  // check if Custom(AddV2) exists before the pass
  auto addV2_out = dynamic_cast<luci::CircleCustomOut *>(_g.output(0)->from());
  EXPECT_NE(nullptr, addV2_out);
  auto addV2 = dynamic_cast<luci::CircleCustom *>(addV2_out->input());
  EXPECT_NE(nullptr, addV2);
  EXPECT_TRUE("AddV2" == addV2->custom_code());

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(true, ret);

  // check if Custom(AddV2) is converted to Add
  auto add = dynamic_cast<luci::CircleAdd *>(_g.output(0)->from());
  EXPECT_NE(nullptr, add);

  // check if BroadcastTo is removed
  auto input_y = dynamic_cast<luci::CircleConst *>(add->y());
  EXPECT_NE(nullptr, input_y);
}

TEST_F(ResolveCustomOpAddPassTest, wrong_custom_code_NEG)
{
  _g.init(false);

  _g.addV2()->custom_code("UNSUPORTED_CUSTOM_CODE");

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}

TEST_F(ResolveCustomOpAddPassTest, wrong_input_type_NEG)
{
  _g.init(false);

  _g.builtin_broadcastTo()->dtype(loco::DataType::BOOL);

  auto ret = _pass.run(_g.g());
  EXPECT_EQ(false, ret);
}
