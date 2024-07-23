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

#include "ForwardPreShiftPass.h"
#include "Support.Cast.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using namespace fme_apply;

namespace
{

/**
 *  PreShift-Slice graphlet
 *
 *   [PreShift]
 *       |
 *     [Slice]
 *
 */
class PreShiftSliceGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _preshift = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _preshift->dtype(loco::DataType::FLOAT32);
    _preshift->shape({1, 4, 4, 16});
    _preshift->custom_code("PreShift");
    _preshift->name("preshift");

    _slice = g->nodes()->create<luci::CircleSlice>();
    _slice->input(_preshift);
    _slice->dtype(loco::DataType::FLOAT32);
    _slice->shape({1, 2, 2, 16});
    _slice->name("slice");
  }

public:
  luci::CircleCustom *_preshift = nullptr;
  luci::CircleSlice *_slice = nullptr;
};

class PreShiftSliceGraph : public luci::test::TestIOGraph, public PreShiftSliceGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 2, 2, 16});
    PreShiftSliceGraphlet::init(g());

    _preshift->inputs(0, input());

    output()->from(_slice);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

} // namespace

TEST(ForwardPreShiftPassTest, preshift_slice)
{
  PreShiftSliceGraph g;
  g.init();

  ForwardPreShiftPass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto pre = to_pre_shift(g.output()->from());
  EXPECT_NE(nullptr, pre);

  auto slice = dynamic_cast<luci::CircleSlice *>(pre->inputs(0));
  EXPECT_NE(nullptr, slice);

  EXPECT_EQ(4, pre->rank());
  EXPECT_EQ(1, pre->dim(0).value());
  EXPECT_EQ(2, pre->dim(1).value());
  EXPECT_EQ(2, pre->dim(2).value());
  EXPECT_EQ(16, pre->dim(3).value());
}

TEST(ForwardPreShiftPassTest, preshift_conv_NEG)
{
  PreShiftSliceGraph g;
  g.init();

  // Replace Pad with Conv2D
  auto conv = g.g()->nodes()->create<luci::CircleConv2D>();
  conv->input(g._preshift);
  g.output()->from(conv);

  ForwardPreShiftPass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}
