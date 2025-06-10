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

#include "InsertScaleShift.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using namespace fme_apply;

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

luci::CircleCustom *to_scale(loco::Node *node)
{
  auto n = dynamic_cast<luci::CircleCustom *>(node);
  if (not n)
    return nullptr;

  if (n->custom_code() != "scale")
    return nullptr;

  return n;
}

} // namespace

TEST(InsertScaleShiftTest, scale_only)
{
  std::vector<EqualizePattern> p;
  EqualizePattern pattern;
  {
    pattern.front = "conv1";
    pattern.back = "conv2";
    pattern.type = EqualizePattern::Type::ScaleOnly;
    for (uint32_t i = 0; i < 16; i++)
      pattern.act_scale.push_back(1.0);
  }
  p.emplace_back(pattern);

  InsertScaleShift iss(p);

  ConvConvGraph g;
  g.init();

  EXPECT_NO_THROW(iss.run(g.g()));

  auto pre = to_scale(g._conv2->input());
  EXPECT_NE(nullptr, pre);

  auto post = to_scale(pre->inputs(0));
  EXPECT_NE(nullptr, post);
}
