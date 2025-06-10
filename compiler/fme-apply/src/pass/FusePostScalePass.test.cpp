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

#include "FusePostScalePass.h"
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
 *  PostScale-Conv graphlet
 *
 *     [Conv]
 *       |
 *   [PostScale]
 *
 */
class PostScaleConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    std::vector<float> filter_val(3 * 3 * 3 * 3 /* size */, 1.0 /*value */);

    _conv = g->nodes()->create<luci::CircleConv2D>();
    _conv->filter(
      create_const_node(g, loco::DataType::FLOAT32, {3, 3, 3, 3} /* shape */, filter_val));
    _conv->bias(
      create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value*/));
    _conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv->dtype(loco::DataType::FLOAT32);
    _conv->shape({1, 4, 4, 3});
    _conv->padding(luci::Padding::SAME);
    _conv->name("conv");

    _postscale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _postscale->dtype(loco::DataType::FLOAT32);
    _postscale->inputs(0, _conv);
    _postscale->inputs(
      1, create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value */));
    _postscale->shape({1, 4, 4, 3});
    _postscale->custom_code("scale");
    _postscale->name("postscale");
  }

public:
  luci::CircleCustom *_postscale = nullptr;
  luci::CircleConv2D *_conv = nullptr;
};

class PostScaleConvGraph : public luci::test::TestIOGraph, public PostScaleConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 3}, {1, 4, 4, 3});
    PostScaleConvGraphlet::init(g());

    _conv->input(input());

    output()->from(_postscale);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  PostScale-DConv graphlet
 *
 *     [DConv]
 *       |
 *   [PostScale]
 *
 */
class PostScaleDConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    std::vector<float> filter_val(1 * 3 * 3 * 3 /* size */, 1.0 /*value */);

    _dconv = g->nodes()->create<luci::CircleDepthwiseConv2D>();
    _dconv->filter(
      create_const_node(g, loco::DataType::FLOAT32, {1, 3, 3, 3} /* shape */, filter_val));
    _dconv->bias(
      create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value*/));
    _dconv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _dconv->dtype(loco::DataType::FLOAT32);
    _dconv->shape({1, 4, 4, 3});
    _dconv->padding(luci::Padding::SAME);
    _dconv->name("dconv");

    _postscale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _postscale->dtype(loco::DataType::FLOAT32);
    _postscale->inputs(0, _dconv);
    _postscale->inputs(
      1, create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value */));
    _postscale->shape({1, 4, 4, 3});
    _postscale->custom_code("scale");
    _postscale->name("postscale");
  }

public:
  luci::CircleCustom *_postscale = nullptr;
  luci::CircleDepthwiseConv2D *_dconv = nullptr;
};

class PostScaleDConvGraph : public luci::test::TestIOGraph, public PostScaleDConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 3}, {1, 4, 4, 3});
    PostScaleDConvGraphlet::init(g());

    _dconv->input(input());

    output()->from(_postscale);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  PostScale-TConv graphlet
 *
 *     [TConv]
 *       |
 *   [PostScale]
 *
 */
class PostScaleTConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    std::vector<float> filter_val(3 * 3 * 3 * 3 /* size */, 1.0 /*value */);

    _tconv = g->nodes()->create<luci::CircleTransposeConv>();
    _tconv->filter(
      create_const_node(g, loco::DataType::FLOAT32, {3, 3, 3, 3} /* shape */, filter_val));
    _tconv->bias(
      create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {1, 1, 1} /* value*/));
    _tconv->dtype(loco::DataType::FLOAT32);
    _tconv->shape({1, 4, 4, 3});
    _tconv->padding(luci::Padding::SAME);
    _tconv->name("conv");

    _postscale = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _postscale->dtype(loco::DataType::FLOAT32);
    _postscale->inputs(0, _tconv);
    _postscale->inputs(
      1, create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value */));
    _postscale->shape({1, 4, 4, 3});
    _postscale->custom_code("scale");
    _postscale->name("postscale");
  }

public:
  luci::CircleCustom *_postscale = nullptr;
  luci::CircleTransposeConv *_tconv = nullptr;
};

class PostScaleTConvGraph : public luci::test::TestIOGraph, public PostScaleTConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 3}, {1, 4, 4, 3});
    PostScaleTConvGraphlet::init(g());

    _tconv->outBackprop(input());

    output()->from(_postscale);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

} // namespace

TEST(FusePostScalePassTest, postscale_conv)
{
  PostScaleConvGraph g;
  g.init();

  FusePostScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto conv = dynamic_cast<luci::CircleConv2D *>(g.output()->from());
  EXPECT_NE(nullptr, conv);

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

TEST(FusePostScalePassTest, postscale_conv_NEG)
{
  PostScaleConvGraph g;
  g.init();
  g._conv->fusedActivationFunction(luci::FusedActFunc::RELU6);

  FusePostScalePass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}

TEST(FusePostScalePassTest, postscale_dconv)
{
  PostScaleDConvGraph g;
  g.init();

  FusePostScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto dconv = dynamic_cast<luci::CircleDepthwiseConv2D *>(g.output()->from());
  EXPECT_NE(nullptr, dconv);

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

TEST(FusePostScalePassTest, postscale_dconv_NEG)
{
  PostScaleDConvGraph g;
  g.init();
  g._dconv->fusedActivationFunction(luci::FusedActFunc::RELU6);

  FusePostScalePass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}

TEST(FusePostScalePassTest, postscale_tconv)
{
  PostScaleTConvGraph g;
  g.init();

  FusePostScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto tconv = dynamic_cast<luci::CircleTransposeConv *>(g.output()->from());
  EXPECT_NE(nullptr, tconv);

  // Check weights
  auto w = dynamic_cast<luci::CircleConst *>(tconv->filter());
  EXPECT_NE(nullptr, w);
  EXPECT_EQ(loco::DataType::FLOAT32, w->dtype());
  EXPECT_EQ(3 * 3 * 3 * 3, w->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < 3 * 3 * 3 * 3; i++)
  {
    EXPECT_FLOAT_EQ(2.0, w->at<loco::DataType::FLOAT32>(i));
  }

  // Check bias
  auto b = dynamic_cast<luci::CircleConst *>(tconv->bias());
  EXPECT_NE(nullptr, b);
  EXPECT_EQ(loco::DataType::FLOAT32, b->dtype());
  EXPECT_EQ(3, b->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < 3; i++)
  {
    EXPECT_FLOAT_EQ(2.0, b->at<loco::DataType::FLOAT32>(i));
  }
}

TEST(FusePostScalePassTest, postscale_tconv_no_bias)
{
  PostScaleTConvGraph g;
  g.init();

  auto no_bias = g.g()->nodes()->create<luci::CircleOutputExclude>();
  g._tconv->bias(no_bias);

  FusePostScalePass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto tconv = dynamic_cast<luci::CircleTransposeConv *>(g.output()->from());
  EXPECT_NE(nullptr, tconv);

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

TEST(FusePostScalePassTest, postscale_tconv_NEG)
{
  PostScaleTConvGraph g;
  g.init();

  g._postscale->inputs(0, g.input());
  g.output()->from(g._tconv);

  FusePostScalePass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}
