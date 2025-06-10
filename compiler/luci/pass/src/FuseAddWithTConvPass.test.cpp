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

#include "luci/Pass/FuseAddWithTConvPass.h"

#include "helpers/CreateCircleConst.h"

#include <luci/IR/CircleNodes.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

#define ADD_VAL 5.0f
namespace
{

using namespace luci::test;

/**
 *  Graph for this test
 *
 *  BEFORE (without extra_successor)
 *
 *                            |
 *   [CircleConst]  [CircleTransposeConv]
 *               \     |
 *          [CircleAdd w/ Relu]
 *                  |
 *
 *  BEFORE (with extra_successor)
 *
 *                            |
 *   [CircleConst]  [CircleTransposeConv]
 *               \     |             |
 *          [CircleAdd w/ Relu]  [extra FC]
 *                  |                |
 *
 *  AFTER (if pass was successful)
 *
 *                           |
 *   [CircleConst as bias]   |
 *             \             |
 *            [CircleTransposeConv]
 *                      |
 *             ([CircleRelu/Relu])
 *                      |
 *
 */
class TConvAddGraphlet
{
public:
  void init(loco::Graph *g, luci::FusedActFunc tconv_activation, bool use_bias,
            bool extra_successor)
  {
    _tconv = g->nodes()->create<luci::CircleTransposeConv>();

    std::vector<float> input_sizes_val = {1, 4, 4, 1};
    _tconv_i = luci::create_const_node(g, loco::DataType::FLOAT32, {4}, input_sizes_val);
    _tconv->inputSizes(_tconv_i);

    std::vector<float> filter_val(18);
    for (uint32_t i = 0; i < 18; i++)
      filter_val.at(i) = i;

    _tconv_f = luci::create_const_node(g, loco::DataType::FLOAT32, {1, 3, 3, 2}, filter_val);
    _tconv->filter(_tconv_f);

    if (use_bias)
    {
      std::vector<float> bias_val(1, 3.0f);
      _tconv_b = luci::create_const_node(g, loco::DataType::FLOAT32, {1}, bias_val);
    }
    else
    {
      // Create CircleOutputExclude -- no bias
      _tconv_b = g->nodes()->create<luci::CircleOutputExclude>();
    }
    _tconv->bias(_tconv_b);

    _tconv->padding(luci::Padding::VALID);
    auto _stride = _tconv->stride();
    _stride->w(1);
    _stride->h(1);
    _tconv->fusedActivationFunction(tconv_activation);
    _tconv->dtype(loco::DataType::FLOAT32);
    _tconv->shape({1, 4, 4, 1});
    _tconv->name("tconv");

    if (extra_successor)
    {
      _extra_succ = g->nodes()->create<luci::CircleFullyConnected>();
      // Set previous TConv as input to bump number of successors for it:
      _extra_succ->input(_tconv);
      std::vector<float> weights_val(8);
      _extra_f = luci::create_const_node(g, loco::DataType::FLOAT32, {1, 8}, weights_val);
      _extra_succ->weights(_extra_f);
      _extra_succ->bias(nullptr);
      _extra_succ->fusedActivationFunction(luci::FusedActFunc::NONE);
      _extra_succ->dtype(loco::DataType::FLOAT32);
      _extra_succ->shape({1, 4, 4, 1});
      _extra_succ->name("extra_fc");
    }

    std::vector<float> add_values(1, ADD_VAL);
    _add_c = luci::create_const_node(g, loco::DataType::FLOAT32, {1}, add_values);
    _add_c->name("const_c");

    _add = g->nodes()->create<luci::CircleAdd>();
    _add->x(_tconv);
    _add->y(_add_c);
    _add->fusedActivationFunction(luci::FusedActFunc::RELU);
    _add->dtype(loco::DataType::FLOAT32);
    _add->shape({1, 4, 4, 1});

    _add->name("add");
  }

protected:
  luci::CircleTransposeConv *_tconv = nullptr;
  luci::CircleConst *_tconv_i = nullptr;
  luci::CircleConst *_tconv_f = nullptr;
  luci::CircleNode *_tconv_b = nullptr;
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_add_c = nullptr;
  luci::CircleFullyConnected *_extra_succ = nullptr;
  luci::CircleConst *_extra_f = nullptr;
};

class FuseAddWithTConvTestGraph : public TestIOGraph, public TConvAddGraphlet
{
public:
  void init(luci::FusedActFunc tconv_activation, bool use_bias, bool extra_successor)
  {
    TestIOGraph::init({1, 2, 2, 2}, {1, 4, 4, 1});
    TConvAddGraphlet::init(g(), tconv_activation, use_bias, extra_successor);

    _tconv->outBackprop(input());

    output()->from(_add);
  }
};

class FuseAddWithTConvPassTest : public ::testing::Test
{
public:
  FuseAddWithTConvTestGraph g;
  luci::FuseAddWithTConvPass pass;
};

} // namespace

TEST_F(FuseAddWithTConvPassTest, tconv_add_fuse)
{
  g.init(luci::FusedActFunc::NONE, false /* use_bias */, false /* extra_successor */);

  EXPECT_EQ(true, pass.run(g.g()));

  auto relu = dynamic_cast<luci::CircleRelu *>(g.output()->from());
  EXPECT_NE(nullptr, relu);
  EXPECT_STREQ(relu->name().c_str(), "const_c/Relu");

  auto tconv = dynamic_cast<luci::CircleTransposeConv *>(relu->features());
  EXPECT_NE(nullptr, tconv);

  auto bias = loco::must_cast<luci::CircleConst *>(tconv->bias());
  EXPECT_NE(nullptr, bias);

  for (uint32_t i = 0; i < bias->size<loco::DataType::FLOAT32>(); i++)
  {
    EXPECT_EQ(ADD_VAL, bias->at<loco::DataType::FLOAT32>(i));
  }
}

TEST_F(FuseAddWithTConvPassTest, tconv_with_bias_NEG)
{
  g.init(luci::FusedActFunc::NONE, true /* use_bias */, false /* extra_successor */);

  EXPECT_EQ(false, pass.run(g.g()));
}

TEST_F(FuseAddWithTConvPassTest, tconv_with_activation_NEG)
{
  g.init(luci::FusedActFunc::RELU, false /* use_bias */, false /* extra_successor */);

  EXPECT_EQ(false, pass.run(g.g()));
}

TEST_F(FuseAddWithTConvPassTest, tconv_with_extra_successor_NEG)
{
  g.init(luci::FusedActFunc::NONE, false /* use_bias */, true /* extra_successor */);

  EXPECT_EQ(false, pass.run(g.g()));
}

TEST_F(FuseAddWithTConvPassTest, name)
{
  luci::FuseAddWithTConvPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}
