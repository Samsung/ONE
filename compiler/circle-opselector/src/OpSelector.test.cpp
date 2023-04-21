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

#include "OpSelector.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Conv-Donv graphlet
 *
 *   [Conv]
 *      |
 *   [Donv]
 *
 */
class ConvDonvGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _conv_filter = g->nodes()->create<luci::CircleConst>();
    _conv_filter->dtype(loco::DataType::FLOAT32);
    _conv_filter->shape({16, 1, 1, 16});
    _conv_filter->name("conv_filter");

    _conv_bias = g->nodes()->create<luci::CircleConst>();
    _conv_bias->dtype(loco::DataType::FLOAT32);
    _conv_bias->shape({16});
    _conv_bias->name("conv_bias");

    _conv = g->nodes()->create<luci::CircleConv2D>();
    _conv->padding(luci::Padding::SAME);
    _conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv->dtype(loco::DataType::FLOAT32);
    _conv->shape({1, 4, 4, 16});
    _conv->name("conv");
    _conv->filter(_conv_filter);
    _conv->bias(_conv_bias);

    _dconv_filter = g->nodes()->create<luci::CircleConst>();
    _dconv_filter->dtype(loco::DataType::FLOAT32);
    _dconv_filter->shape({16, 1, 1, 16});
    _dconv_filter->name("dconv_filter");

    _dconv_bias = g->nodes()->create<luci::CircleConst>();
    _dconv_bias->dtype(loco::DataType::FLOAT32);
    _dconv_bias->shape({16});
    _dconv_bias->name("dconv_bias");

    _dconv = g->nodes()->create<luci::CircleDepthwiseConv2D>();
    _dconv->input(_conv);
    _dconv->depthMultiplier(1);
    _dconv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _dconv->dtype(loco::DataType::FLOAT32);
    _dconv->shape({1, 4, 4, 16});
    _dconv->padding(luci::Padding::SAME);
    _dconv->name("dconv");
    _dconv->filter(_dconv_filter);
    _dconv->bias(_dconv_bias);
  }

protected:
  luci::CircleConv2D *_conv{nullptr};
  luci::CircleConst *_conv_filter{nullptr};
  luci::CircleConst *_conv_bias{nullptr};
  luci::CircleDepthwiseConv2D *_dconv{nullptr};
  luci::CircleConst *_dconv_filter{nullptr};
  luci::CircleConst *_dconv_bias{nullptr};
};

class ConvDonvGraph : public luci::test::TestIOGraph, public ConvDonvGraphlet
{
public:
  ConvDonvGraph()
  {
    luci::test::TestIOGraph::init({1, 4, 4, 16}, {1, 4, 4, 16});
    ConvDonvGraphlet::init(g());

    _conv->input(input());

    output()->from(_dconv);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

} // namespace

TEST(OpSelectorTest, select_by_name)
{
  auto m = luci::make_module();

  ConvDonvGraph g;
  g.transfer_to(m.get());

  opselector::OpSelector op_selector{m.get()};

  // Select conv only
  auto conv_module =
    op_selector.select_by<opselector::SelectType::NAME>(std::vector<std::string>{"conv"});
  ASSERT_EQ(1, conv_module->size());

  auto conv_graph = conv_module->graph(0);
  ASSERT_EQ(1, conv_graph->outputs()->size());

  auto output_node1 = luci::output_node(conv_graph, 0);
  auto conv = loco::must_cast<luci::CircleConv2D *>(output_node1->from());
  EXPECT_STREQ("conv", conv->name().c_str());
  auto conv_filter = loco::must_cast<luci::CircleConst *>(conv->filter());
  EXPECT_STREQ("conv_filter", conv_filter->name().c_str());
  auto conv_bias = loco::must_cast<luci::CircleConst *>(conv->bias());
  EXPECT_STREQ("conv_bias", conv_bias->name().c_str());

  // Select dconv only
  auto dconv_module =
    op_selector.select_by<opselector::SelectType::NAME>(std::vector<std::string>{"dconv"});
  ASSERT_EQ(1, dconv_module->size());

  auto dconv_graph = dconv_module->graph(0);
  ASSERT_EQ(1, dconv_graph->outputs()->size());

  auto output_node2 = luci::output_node(dconv_graph, 0);
  auto dconv = loco::must_cast<luci::CircleDepthwiseConv2D *>(output_node2->from());
  EXPECT_STREQ("dconv", dconv->name().c_str());
  auto dconv_filter = loco::must_cast<luci::CircleConst *>(dconv->filter());
  EXPECT_STREQ("dconv_filter", dconv_filter->name().c_str());
  auto dconv_bias = loco::must_cast<luci::CircleConst *>(dconv->bias());
  EXPECT_STREQ("dconv_bias", dconv_bias->name().c_str());
}

TEST(OpSelectorTest, select_by_name_NEG)
{
  auto m = luci::make_module();

  ConvDonvGraph g;
  g.transfer_to(m.get());

  opselector::OpSelector op_selector{m.get()};

  EXPECT_ANY_THROW(
    op_selector.select_by<opselector::SelectType::NAME>(std::vector<std::string>{","}));
}
