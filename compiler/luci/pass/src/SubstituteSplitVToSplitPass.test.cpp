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

#include "luci/Pass/SubstituteSplitVToSplitPass.h"

#include "helpers/CreateCircleConst.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

const int N = 1;
const int C = 32;
const int H = 8;
const int W = 8;

/**
 *  graph having SplitV operator
 *
 *                [CircleInput]
 *                      |
 *                [CircleSplitV]
 *                     /  \
 *      [CircleSplitVOut] [CircleSplitVOut]
 *             |                   |
 *       [CircleOutput]     [CircleOutput]
 */
class SplitVGraphlet
{
public:
  SplitVGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    const std::vector<int32_t> splits{16, 16};
    auto size_splits = luci::create_const_node(g, loco::DataType::S32, {2}, splits);

    const std::vector<int32_t> dim{3};
    auto split_dim = luci::create_const_node(g, loco::DataType::S32, {1}, dim);

    _sv = g->nodes()->create<luci::CircleSplitV>();
    _sv->size_splits(size_splits);
    _sv->split_dim(split_dim);
    _sv->num_split(2);
    _sv->name("SplitV");

    _svo1 = g->nodes()->create<luci::CircleSplitVOut>();
    _svo1->input(_sv);
    _svo1->index(0);
    _svo1->name("SplitV0");

    _svo2 = g->nodes()->create<luci::CircleSplitVOut>();
    _svo2->input(_sv);
    _svo2->index(1);
    _svo2->name("SplitV1");
  }

public:
  luci::CircleSplitV *split_v() { return _sv; }
  luci::CircleSplitVOut *split_vo1() { return _svo1; }
  luci::CircleSplitVOut *split_vo2() { return _svo2; }

protected:
  luci::CircleSplitV *_sv = nullptr;
  luci::CircleSplitVOut *_svo1 = nullptr;
  luci::CircleSplitVOut *_svo2 = nullptr;
};

class SplitVGraph : public TestIsGraphlet<1>, public TestOsGraphlet<2>, public SplitVGraphlet
{
public:
  SplitVGraph() = default;

  void init(void)
  {
    TestIsGraphlet<1>::init(g(), {{N, C, H, W}});
    TestOsGraphlet<2>::init(g(), {{N, C, H / 2, W / 2}, {N, C, H / 2, W / 2}});
    SplitVGraphlet::init(g());

    split_v()->input(input(0));

    output(0)->from(split_vo1());
    output(1)->from(split_vo2());
  }
};

class SubstituteSplitVToSplitPassTest : public ::testing::Test
{
public:
  SplitVGraph g;
  luci::SubstituteSplitVToSplitPass pass;
};

} // namespace

/**
 *  Optimized graph looks like below.
 *
 *                [CircleInput]
 *                      |
 *                [CircleSplit]
 *                     /  \
 *      [CircleSplitOut] [CircleSplitOut]
 *             |                 |
 *       [CircleOutput]   [CircleOutput]
 */
TEST_F(SubstituteSplitVToSplitPassTest, simple_test)
{
  g.init();

  auto ret = pass.run(g.g());
  EXPECT_EQ(true, ret);

  auto so1 = dynamic_cast<luci::CircleSplitOut *>(g.output(0)->from());
  EXPECT_NE(nullptr, so1);

  auto so2 = dynamic_cast<luci::CircleSplitOut *>(g.output(1)->from());
  EXPECT_NE(nullptr, so2);

  EXPECT_EQ(so1->input(), so2->input());

  auto s = dynamic_cast<luci::CircleSplit *>(so1->input());
  EXPECT_NE(nullptr, s);

  auto input = dynamic_cast<luci::CircleInput *>(s->input());
  EXPECT_NE(nullptr, input);
}

TEST_F(SubstituteSplitVToSplitPassTest, wrong_condition_NEG)
{
  g.init();

  g.split_v()->num_split(3); // Wrong num_split
  auto ret = pass.run(g.g());

  EXPECT_EQ(false, ret);
}
