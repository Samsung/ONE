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

#include "FusePreScalePass.h"
#include "Support.Cast.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using namespace fme_apply;

namespace
{

luci::CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<float> &values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape[i];
    size *= shape[i];
  }
  node->shape_status(luci::ShapeStatus::VALID);

  assert(values.size() == size); // FIX_CALLER_UNLESS

  node->size<loco::DataType::FLOAT32>(size);
  for (uint32_t i = 0; i < values.size(); ++i)
    node->at<loco::DataType::FLOAT32>(i) = values[i];

  return node;
}

/**
 *  PreScale-Conv graphlet
 *
 *   [PreScale]
 *       |
 *     [Conv]
 *
 */
class PreScaleConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _prescale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _prescale->dtype(loco::DataType::FLOAT32);
    _prescale->inputs(
      1, create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value */));
    _prescale->shape({1, 4, 4, 3});
    _prescale->custom_code("scale");
    _prescale->name("prescale");

    std::vector<float> filter_val(3 * 3 * 3 * 3 /* size */, 1.0 /*value */);

    _conv = g->nodes()->create<luci::CircleConv2D>();
    _conv->input(_prescale);
    _conv->filter(
      create_const_node(g, loco::DataType::FLOAT32, {3, 3, 3, 3} /* shape */, filter_val));
    _conv->bias(
      create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value*/));
    _conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv->dtype(loco::DataType::FLOAT32);
    _conv->shape({1, 4, 4, 3});
    _conv->padding(luci::Padding::SAME);
    _conv->name("conv");
  }

public:
  luci::CircleCustom *_prescale = nullptr;
  luci::CircleConv2D *_conv = nullptr;
};

class PreScaleConvGraph : public luci::test::TestIOGraph, public PreScaleConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 3}, {1, 4, 4, 3});
    PreScaleConvGraphlet::init(g());

    _prescale->inputs(0, input());

    output()->from(_conv);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  PreScale-TConv graphlet
 *
 *   [PreScale]
 *       |
 *     [TConv]
 *
 */
class PreScaleTConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _prescale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _prescale->dtype(loco::DataType::FLOAT32);
    _prescale->inputs(
      1, create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value */));
    _prescale->shape({1, 4, 4, 3});
    _prescale->custom_code("scale");
    _prescale->name("prescale");

    std::vector<float> filter_val(3 * 3 * 3 * 3 /* size */, 1.0 /*value */);

    _tconv = g->nodes()->create<luci::CircleTransposeConv>();
    _tconv->outBackprop(_prescale);
    _tconv->filter(
      create_const_node(g, loco::DataType::FLOAT32, {3, 3, 3, 3} /* shape */, filter_val));
    _tconv->bias(
      create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value*/));
    _tconv->dtype(loco::DataType::FLOAT32);
    _tconv->shape({1, 4, 4, 3});
    _tconv->padding(luci::Padding::SAME);
    _tconv->name("tconv");
  }

public:
  luci::CircleCustom *_prescale = nullptr;
  luci::CircleTransposeConv *_tconv = nullptr;
};

class PreScaleTConvGraph : public luci::test::TestIOGraph, public PreScaleTConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 3}, {1, 4, 4, 3});
    PreScaleTConvGraphlet::init(g());

    _prescale->inputs(0, input());

    output()->from(_tconv);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  PreScale-DConv graphlet
 *
 *   [PreScale]
 *       |
 *     [DConv]
 *
 */
class PreScaleDConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _prescale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _prescale->dtype(loco::DataType::FLOAT32);
    _prescale->inputs(
      1, create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value */));
    _prescale->shape({1, 4, 4, 3});
    _prescale->custom_code("scale");
    _prescale->name("prescale");

    std::vector<float> filter_val(1 * 3 * 3 * 3 /* size */, 1.0 /*value */);

    _dconv = g->nodes()->create<luci::CircleDepthwiseConv2D>();
    _dconv->input(_prescale);
    _dconv->filter(
      create_const_node(g, loco::DataType::FLOAT32, {1, 3, 3, 3} /* shape */, filter_val));
    _dconv->bias(
      create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value*/));
    _dconv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _dconv->depthMultiplier(1);
    _dconv->dtype(loco::DataType::FLOAT32);
    _dconv->shape({1, 4, 4, 3});
    _dconv->padding(luci::Padding::SAME);
    _dconv->name("dconv");
  }

public:
  luci::CircleCustom *_prescale = nullptr;
  luci::CircleDepthwiseConv2D *_dconv = nullptr;
};

class PreScaleDConvGraph : public luci::test::TestIOGraph, public PreScaleDConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 3}, {1, 4, 4, 3});
    PreScaleDConvGraphlet::init(g());

    _prescale->inputs(0, input());

    output()->from(_dconv);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

} // namespace

TEST(FusePreScalePassTest, prescale_conv)
{
  PreScaleConvGraph g;
  g.init();

  FusePreScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto conv = dynamic_cast<luci::CircleConv2D *>(g.output()->from());
  EXPECT_NE(nullptr, conv);

  auto pre_scale = to_scale(conv->input());
  EXPECT_EQ(nullptr, pre_scale); // No pre_scale

  // Check weights
  auto w = dynamic_cast<luci::CircleConst *>(conv->filter());
  EXPECT_NE(nullptr, w);
  EXPECT_EQ(loco::DataType::FLOAT32, w->dtype());
  EXPECT_EQ(3 * 3 * 3 * 3, w->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < 3 * 3 * 3 * 3; i++)
  {
    EXPECT_FLOAT_EQ(2.0, w->at<loco::DataType::FLOAT32>(i));
  }
}

TEST(FusePreScalePassTest, prescale_conv_NEG)
{
  PreScaleConvGraph g;
  g.init();
  g._conv->input(g.input());

  FusePreScalePass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}

TEST(FusePreScalePassTest, prescale_tconv)
{
  PreScaleTConvGraph g;
  g.init();

  FusePreScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto tconv = dynamic_cast<luci::CircleTransposeConv *>(g.output()->from());
  EXPECT_NE(nullptr, tconv);

  auto pre_scale = to_scale(tconv->outBackprop());
  EXPECT_EQ(nullptr, pre_scale); // No pre_scale

  // Check weights
  auto w = dynamic_cast<luci::CircleConst *>(tconv->filter());
  EXPECT_NE(nullptr, w);
  EXPECT_EQ(loco::DataType::FLOAT32, w->dtype());
  EXPECT_EQ(3 * 3 * 3 * 3, w->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < 3 * 3 * 3 * 3; i++)
  {
    EXPECT_FLOAT_EQ(2.0, w->at<loco::DataType::FLOAT32>(i));
  }
}

TEST(FusePreScalePassTest, prescale_tconv_NEG)
{
  PreScaleTConvGraph g;
  g.init();
  g._tconv->outBackprop(g.input());

  FusePreScalePass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}

TEST(FusePreScalePassTest, prescale_dconv)
{
  PreScaleDConvGraph g;
  g.init();

  FusePreScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto dconv = dynamic_cast<luci::CircleDepthwiseConv2D *>(g.output()->from());
  EXPECT_NE(nullptr, dconv);

  auto pre_scale = to_scale(dconv->input());
  EXPECT_EQ(nullptr, pre_scale); // No pre_scale

  // Check weights
  auto w = dynamic_cast<luci::CircleConst *>(dconv->filter());
  EXPECT_NE(nullptr, w);
  EXPECT_EQ(loco::DataType::FLOAT32, w->dtype());
  EXPECT_EQ(1 * 3 * 3 * 3, w->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < 1 * 3 * 3 * 3; i++)
  {
    EXPECT_FLOAT_EQ(2.0, w->at<loco::DataType::FLOAT32>(i));
  }
}

TEST(FusePreScalePassTest, prescale_dconv_NEG)
{
  PreScaleDConvGraph g;
  g.init();
  g._dconv->input(g.input());

  FusePreScalePass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}
