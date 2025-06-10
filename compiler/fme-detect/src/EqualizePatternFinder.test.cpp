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

#include "EqualizePatternFinder.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using namespace fme_detect;

namespace
{

/**
 *  Conv-Conv graphlet
 *
 *   [Conv]
 *      |
 *   [Conv]
 *
 */
class ConvConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _conv1 = g->nodes()->create<luci::CircleConv2D>();
    _conv1->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv1->dtype(loco::DataType::FLOAT32);
    _conv1->shape({1, 4, 4, 16});
    _conv1->padding(luci::Padding::SAME);
    _conv1->name("conv1");

    _conv2 = g->nodes()->create<luci::CircleConv2D>();
    _conv2->input(_conv1);
    _conv2->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv2->dtype(loco::DataType::FLOAT32);
    _conv2->shape({1, 4, 4, 16});
    _conv1->padding(luci::Padding::SAME);
    _conv2->name("conv2");
  }

public:
  luci::CircleConv2D *_conv1 = nullptr;
  luci::CircleConv2D *_conv2 = nullptr;
};

class ConvConvGraph : public luci::test::TestIOGraph, public ConvConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 4, 4, 16});
    ConvConvGraphlet::init(g());

    _conv1->input(input());

    output()->from(_conv2);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  Conv-Pad-Conv graphlet
 *
 *   [Conv]
 *      |
 *    [Pad]
 *      |
 *   [Conv]
 *
 */
class ConvPadConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _conv1 = g->nodes()->create<luci::CircleConv2D>();
    _conv1->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv1->dtype(loco::DataType::FLOAT32);
    _conv1->shape({1, 4, 4, 16});
    _conv1->padding(luci::Padding::SAME);
    _conv1->name("conv1");

    _pad = g->nodes()->create<luci::CirclePad>();
    _pad->input(_conv1);
    _pad->dtype(loco::DataType::FLOAT32);
    _pad->name("pad");

    _conv2 = g->nodes()->create<luci::CircleConv2D>();
    _conv2->input(_pad);
    _conv2->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv2->dtype(loco::DataType::FLOAT32);
    _conv2->shape({1, 4, 4, 16});
    _conv2->padding(luci::Padding::VALID);
    _conv2->name("conv2");
  }

public:
  luci::CircleConv2D *_conv1 = nullptr;
  luci::CirclePad *_pad = nullptr;
  luci::CircleConv2D *_conv2 = nullptr;
};

class ConvPadConvGraph : public luci::test::TestIOGraph, public ConvPadConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 4, 4, 16});
    ConvPadConvGraphlet::init(g());

    _conv1->input(input());

    output()->from(_conv2);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  Conv-Maxpool2D-Conv graphlet
 *
 *   [Conv]
 *      |
 * [Maxpool2D]
 *      |
 *   [Conv]
 *
 */
class ConvMaxpoolConvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _conv1 = g->nodes()->create<luci::CircleConv2D>();
    _conv1->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv1->dtype(loco::DataType::FLOAT32);
    _conv1->shape({1, 4, 4, 16});
    _conv1->padding(luci::Padding::SAME);
    _conv1->name("conv1");

    _maxpool = g->nodes()->create<luci::CircleMaxPool2D>();
    _maxpool->padding(luci::Padding::SAME);
    _maxpool->value(_conv1);
    _maxpool->dtype(loco::DataType::FLOAT32);
    _maxpool->name("maxpool");

    _conv2 = g->nodes()->create<luci::CircleConv2D>();
    _conv2->input(_maxpool);
    _conv2->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv2->dtype(loco::DataType::FLOAT32);
    _conv2->shape({1, 4, 4, 16});
    _conv2->padding(luci::Padding::SAME);
    _conv2->name("conv2");
  }

public:
  luci::CircleConv2D *_conv1 = nullptr;
  luci::CircleMaxPool2D *_maxpool = nullptr;
  luci::CircleConv2D *_conv2 = nullptr;
};

class ConvMaxpoolConvGraph : public luci::test::TestIOGraph, public ConvMaxpoolConvGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 4, 4, 16});
    ConvMaxpoolConvGraphlet::init(g());

    _conv1->input(input());

    output()->from(_conv2);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  Conv with two sucessors
 *
 *       [Conv]
 *        /  \
 *   [Conv]  [Instnorm]
 *        \  /
 *      [Concat]
 */
class ConvConvINGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _conv1 = g->nodes()->create<luci::CircleConv2D>();
    _conv1->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv1->dtype(loco::DataType::FLOAT32);
    _conv1->shape({1, 4, 4, 16});
    _conv1->padding(luci::Padding::SAME);
    _conv1->name("conv1");

    _conv2 = g->nodes()->create<luci::CircleConv2D>();
    _conv2->input(_conv1);
    _conv2->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv2->dtype(loco::DataType::FLOAT32);
    _conv2->shape({1, 4, 4, 16});
    _conv2->padding(luci::Padding::SAME);
    _conv2->name("conv2");

    _instnorm = g->nodes()->create<luci::CircleInstanceNorm>();
    _instnorm->input(_conv1);
    _instnorm->fusedActivationFunction(luci::FusedActFunc::NONE);
    _instnorm->dtype(loco::DataType::FLOAT32);
    _instnorm->shape({1, 4, 4, 16});
    _instnorm->name("instnorm");

    _concat = g->nodes()->create<luci::CircleConcatenation>(2);
    _concat->values(0, _conv2);
    _concat->values(1, _instnorm);
    _concat->dtype(loco::DataType::FLOAT32);
    _concat->shape({1, 4, 4, 32});
    _concat->axis(3);
    _concat->name("concat");
  }

public:
  luci::CircleConv2D *_conv1 = nullptr;
  luci::CircleConv2D *_conv2 = nullptr;
  luci::CircleInstanceNorm *_instnorm = nullptr;
  luci::CircleConcatenation *_concat = nullptr;
};

class ConvConvINGraph : public luci::test::TestIOGraph, public ConvConvINGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 4, 4, 16});
    ConvConvINGraphlet::init(g());

    _conv1->input(input());

    output()->from(_concat);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

/**
 *  TConv-Slice-Instnorm graphlet
 *
 *   [TConv]
 *      |
 *   [Slice]
 *      |
 *   [Instnorm]
 *
 */
class TConvSliceInstnormGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _tconv = g->nodes()->create<luci::CircleTransposeConv>();
    _tconv->dtype(loco::DataType::FLOAT32);
    _tconv->shape({1, 4, 4, 16});
    _tconv->padding(luci::Padding::VALID);
    _tconv->name("tconv");

    _slice = g->nodes()->create<luci::CircleSlice>();
    _slice->input(_tconv);
    _slice->dtype(loco::DataType::FLOAT32);
    _slice->name("slice");

    _instnorm = g->nodes()->create<luci::CircleInstanceNorm>();
    _instnorm->input(_slice);
    _instnorm->fusedActivationFunction(luci::FusedActFunc::NONE);
    _instnorm->dtype(loco::DataType::FLOAT32);
    _instnorm->shape({1, 4, 4, 16});
    _instnorm->name("instnorm");
  }

public:
  luci::CircleTransposeConv *_tconv = nullptr;
  luci::CircleSlice *_slice = nullptr;
  luci::CircleInstanceNorm *_instnorm = nullptr;
};

class TConvSliceInstnormGraph : public luci::test::TestIOGraph, public TConvSliceInstnormGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 4, 4, 16});
    TConvSliceInstnormGraphlet::init(g());

    _tconv->outBackprop(input());

    output()->from(_instnorm);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

} // namespace

TEST(EqualizePatternFinderTest, simple)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  loco::Graph g;
  EXPECT_NO_THROW(epf.find(&g));
}

TEST(EqualizePatternFinderTest, null_graph_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  EXPECT_ANY_THROW(epf.find(nullptr));
}

TEST(EqualizePatternFinderTest, conv_conv)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvConvGraph g;
  g.init();

  auto res = epf.find(g.g());

  EXPECT_EQ(1, res.size());
  EXPECT_EQ("conv1", res[0].front);
  EXPECT_EQ("conv2", res[0].back);
  EXPECT_EQ(EqualizePattern::Type::ScaleOnly, res[0].type);
}

TEST(EqualizePatternFinderTest, conv_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvConvGraph g;
  g.init();
  g.output()->from(g._conv1);
  g._conv2->input(g.input());

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}

TEST(EqualizePatternFinderTest, conv_relu_conv)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvConvGraph g;
  g.init();
  g._conv1->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto res = epf.find(g.g());

  EXPECT_EQ(1, res.size());
  EXPECT_EQ("conv1", res[0].front);
  EXPECT_EQ("conv2", res[0].back);
  EXPECT_EQ(EqualizePattern::Type::ScaleOnly, res[0].type);
}

TEST(EqualizePatternFinderTest, conv_relu6_conv_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvConvGraph g;
  g.init();
  g._conv1->fusedActivationFunction(luci::FusedActFunc::RELU6);

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}

TEST(EqualizePatternFinderTest, conv_tanh_conv_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvConvGraph g;
  g.init();
  g._conv1->fusedActivationFunction(luci::FusedActFunc::TANH);

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}

TEST(EqualizePatternFinderTest, conv_pad_conv)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvPadConvGraph g;
  g.init();

  auto res = epf.find(g.g());

  EXPECT_EQ(1, res.size());
  EXPECT_EQ("conv1", res[0].front);
  EXPECT_EQ("conv2", res[0].back);
  EXPECT_EQ(EqualizePattern::Type::ScaleOnly, res[0].type);
}

TEST(EqualizePatternFinderTest, conv_pad_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvPadConvGraph g;
  g.init();
  g.output()->from(g._pad);

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}

TEST(EqualizePatternFinderTest, conv_maxpool_conv)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvMaxpoolConvGraph g;
  g.init();

  auto res = epf.find(g.g());

  EXPECT_EQ(1, res.size());
  EXPECT_EQ("conv1", res[0].front);
  EXPECT_EQ("conv2", res[0].back);
  EXPECT_EQ(EqualizePattern::Type::ScaleOnly, res[0].type);
}

TEST(EqualizePatternFinderTest, conv_relu_pad_conv)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvPadConvGraph g;
  g.init();
  g._conv1->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto res = epf.find(g.g());

  EXPECT_EQ(1, res.size());
  EXPECT_EQ("conv1", res[0].front);
  EXPECT_EQ("conv2", res[0].back);
  EXPECT_EQ(EqualizePattern::Type::ScaleOnly, res[0].type);
}

TEST(EqualizePatternFinderTest, conv_relu6_pad_conv_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvPadConvGraph g;
  g.init();
  g._conv1->fusedActivationFunction(luci::FusedActFunc::RELU6);

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}

TEST(EqualizePatternFinderTest, conv_tanh_pad_conv_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  ConvPadConvGraph g;
  g.init();
  g._conv1->fusedActivationFunction(luci::FusedActFunc::TANH);

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}

TEST(EqualizePatternFinderTest, tconv_slice_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = true;
  }
  EqualizePatternFinder epf(ctx);

  TConvSliceInstnormGraph g;
  g.init();
  g.output()->from(g._slice);

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}

TEST(EqualizePatternFinderTest, dup_op_NEG)
{
  EqualizePatternFinder::Context ctx;
  {
    ctx._allow_dup_op = false;
  }
  EqualizePatternFinder epf(ctx);

  ConvConvINGraph g;
  g.init();

  auto res = epf.find(g.g());

  EXPECT_EQ(0, res.size());
}
