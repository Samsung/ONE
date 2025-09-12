/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ResolveCustomOpSplitVPass.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/CircleNodes.h>
#include <gtest/gtest.h>

using namespace luci::test;

namespace
{

/**
 *  graph having Custom operator SplitV
 *
 *        [Input]  [Const] [Const]
 *             \    |    /
 *           [Custom(SplitV)]
 *             /    |       \
 *  [CustomOut] [CustomOut] [CustomOut]
 *       |          |           |
 *   [Output]   [Output]     [Output]
 */
class SplitVGraphlet
{
public:
  SplitVGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    // CircleCustom(SplitV)
    _splitv = g->nodes()->create<luci::CircleCustom>(3, 3);
    _splitv->custom_code("SplitV");
    _splitv->shape({1, 2, 2, 192});
    _splitv->dtype(loco::DataType::FLOAT32);
    _splitv->name("splitv");

    // CircleConst
    auto size_splits = g->nodes()->create<luci::CircleConst>();
    size_splits->dtype(loco::DataType::S64);
    size_splits->shape({3});
    size_splits->size<loco::DataType::S64>(3);
    size_splits->at<loco::DataType::S64>(0) = 32;
    size_splits->at<loco::DataType::S64>(1) = 32;
    size_splits->at<loco::DataType::S64>(2) = 128;

    // CircleConst
    auto split_dim = g->nodes()->create<luci::CircleConst>();
    split_dim->dtype(loco::DataType::S32);
    split_dim->rank(0);
    split_dim->size<loco::DataType::S32>(1);
    split_dim->scalar<loco::DataType::S32>() = 3;

    _splitv->inputs(1, size_splits);
    _splitv->inputs(2, split_dim);

    // CircleCustomOut
    _splitv_out1 = g->nodes()->create<luci::CircleCustomOut>();
    _splitv_out1->shape({1, 2, 2, 32});
    _splitv_out1->dtype(loco::DataType::FLOAT32);
    _splitv_out1->index(0);
    _splitv_out1->input(_splitv);

    // CircleCustomOut
    _splitv_out2 = g->nodes()->create<luci::CircleCustomOut>();
    _splitv_out2->shape({1, 2, 2, 32});
    _splitv_out2->dtype(loco::DataType::FLOAT32);
    _splitv_out2->index(1);
    _splitv_out2->input(_splitv);

    // CircleCustomOut
    _splitv_out3 = g->nodes()->create<luci::CircleCustomOut>();
    _splitv_out3->shape({1, 2, 2, 128});
    _splitv_out3->dtype(loco::DataType::FLOAT32);
    _splitv_out3->index(2);
    _splitv_out3->input(_splitv);
  }

public:
  luci::CircleCustom *splitv() { return _splitv; }

protected:
  luci::CircleCustom *_splitv = nullptr;
  luci::CircleCustomOut *_splitv_out1 = nullptr;
  luci::CircleCustomOut *_splitv_out2 = nullptr;
  luci::CircleCustomOut *_splitv_out3 = nullptr;
};

class SplitVGraph : public TestIGraphlet, public TestOsGraphlet<3>, public SplitVGraphlet
{
public:
  SplitVGraph() = default;

  void init(void)
  {
    TestIGraphlet::init(g(), {1, 2, 2, 192});
    TestOsGraphlet<3>::init(g(), {{1, 2, 2, 32}, {1, 2, 2, 32}, {1, 2, 2, 128}});
    SplitVGraphlet::init(g());

    // connect graph
    _splitv->inputs(0, input());

    output(0)->from(_splitv_out1);
    output(1)->from(_splitv_out2);
    output(2)->from(_splitv_out3);
  }
};

class SplitVGraphTest : public ::testing::Test
{
public:
  SplitVGraph g;
  luci::ResolveCustomOpSplitVPass pass;
};

} // namespace

TEST_F(SplitVGraphTest, simple_test)
{
  g.init();

  auto ret = pass.run(g.g());
  EXPECT_EQ(true, ret);

  auto svo_1 = dynamic_cast<luci::CircleSplitVOut *>(g.output(0)->from());
  EXPECT_NE(nullptr, svo_1);
  auto svo_2 = dynamic_cast<luci::CircleSplitVOut *>(g.output(1)->from());
  EXPECT_NE(nullptr, svo_2);
  auto svo_3 = dynamic_cast<luci::CircleSplitVOut *>(g.output(2)->from());
  EXPECT_NE(nullptr, svo_3);

  auto sv = dynamic_cast<luci::CircleSplitV *>(svo_1->input());
  EXPECT_NE(nullptr, sv);
  sv = dynamic_cast<luci::CircleSplitV *>(svo_2->input());
  EXPECT_NE(nullptr, sv);
  sv = dynamic_cast<luci::CircleSplitV *>(svo_3->input());
  EXPECT_NE(nullptr, sv);

  auto size_splits = luci::must_cast<luci::CircleConst *>(sv->size_splits());
  EXPECT_EQ(loco::DataType::S32, size_splits->dtype());
  EXPECT_EQ(32, size_splits->at<loco::DataType::S32>(0));
  EXPECT_EQ(32, size_splits->at<loco::DataType::S32>(1));
  EXPECT_EQ(128, size_splits->at<loco::DataType::S32>(2));

  auto split_dim = luci::must_cast<luci::CircleConst *>(sv->split_dim());
  EXPECT_EQ(loco::DataType::S32, split_dim->dtype());
  EXPECT_EQ(3, split_dim->scalar<loco::DataType::S32>());
}

TEST_F(SplitVGraphTest, wrong_op_NEG)
{
  g.init();

  g.splitv()->custom_code("AddV2");

  auto ret = pass.run(g.g());
  EXPECT_EQ(false, ret);
}
