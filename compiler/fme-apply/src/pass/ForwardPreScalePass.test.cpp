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

#include "ForwardPreScalePass.h"
#include "Support.Cast.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using namespace fme_apply;

namespace
{

/**
 *  PreScale-Pad graphlet
 *
 *   [PreScale]
 *       |
 *     [Pad]
 *
 */
class PreScalePadGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _prescale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _prescale->dtype(loco::DataType::FLOAT32);
    _prescale->shape({1, 4, 4, 16});
    _prescale->custom_code("PreScale");
    _prescale->name("prescale");

    _pad = g->nodes()->create<luci::CirclePad>();
    _pad->input(_prescale);
    _pad->dtype(loco::DataType::FLOAT32);
    _pad->shape({1, 5, 5, 16});
    _pad->name("pad");
  }

public:
  luci::CircleCustom *_prescale = nullptr;
  luci::CirclePad *_pad = nullptr;
};

class PreScalePadGraph : public luci::test::TestIOGraph, public PreScalePadGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 5, 5, 16});
    PreScalePadGraphlet::init(g());

    _prescale->inputs(0, input());

    output()->from(_pad);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  PreScale-Slice graphlet
 *
 *   [PreScale]
 *       |
 *     [Slice]
 *
 */
class PreScaleSliceGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _prescale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _prescale->dtype(loco::DataType::FLOAT32);
    _prescale->shape({1, 4, 4, 16});
    _prescale->custom_code("PreScale");
    _prescale->name("prescale");

    _slice = g->nodes()->create<luci::CircleSlice>();
    _slice->input(_prescale);
    _slice->dtype(loco::DataType::FLOAT32);
    _slice->shape({1, 2, 2, 16});
    _slice->name("slice");
  }

public:
  luci::CircleCustom *_prescale = nullptr;
  luci::CircleSlice *_slice = nullptr;
};

class PreScaleSliceGraph : public luci::test::TestIOGraph, public PreScaleSliceGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 2, 2, 16});
    PreScaleSliceGraphlet::init(g());

    _prescale->inputs(0, input());

    output()->from(_slice);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

} // namespace

TEST(ForwardPreScalePassTest, prescale_pad)
{
  PreScalePadGraph g;
  g.init();

  ForwardPreScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto pre = to_pre_scale(g.output()->from());
  EXPECT_NE(nullptr, pre);

  auto pad = dynamic_cast<luci::CirclePad *>(pre->inputs(0));
  EXPECT_NE(nullptr, pad);

  EXPECT_EQ(4, pre->rank());
  EXPECT_EQ(1, pre->dim(0).value());
  EXPECT_EQ(5, pre->dim(1).value());
  EXPECT_EQ(5, pre->dim(2).value());
  EXPECT_EQ(16, pre->dim(3).value());
}

TEST(ForwardPreScalePassTest, prescale_slice)
{
  PreScaleSliceGraph g;
  g.init();

  ForwardPreScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto pre = to_pre_scale(g.output()->from());
  EXPECT_NE(nullptr, pre);

  auto slice = dynamic_cast<luci::CircleSlice *>(pre->inputs(0));
  EXPECT_NE(nullptr, slice);

  EXPECT_EQ(4, pre->rank());
  EXPECT_EQ(1, pre->dim(0).value());
  EXPECT_EQ(2, pre->dim(1).value());
  EXPECT_EQ(2, pre->dim(2).value());
  EXPECT_EQ(16, pre->dim(3).value());
}

TEST(ForwardPreScalePassTest, prescale_conv_NEG)
{
  PreScalePadGraph g;
  g.init();

  // Replace Pad with Conv2D
  auto conv = g.g()->nodes()->create<luci::CircleConv2D>();
  conv->input(g._prescale);
  g.output()->from(conv);

  ForwardPreScalePass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}
