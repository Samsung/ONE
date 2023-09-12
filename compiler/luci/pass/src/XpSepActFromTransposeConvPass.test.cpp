/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/XpSepActFromTransposeConvPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>
#include "test/TestFirstNode.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class TrConvGraphlet
{
public:
  TrConvGraphlet() = default;

public:
  void init(loco::Graph *g, ShapeU32 wshape)
  {
    const uint32_t elements_num = num_elements(wshape);

    // trconv inputSizes
    auto wshape_size = static_cast<uint32_t>(wshape.size());
    _inpsize = g->nodes()->create<luci::CircleConst>();
    _inpsize->dtype(loco::DataType::S32);
    _inpsize->shape({wshape_size});
    _inpsize->size<loco::DataType::S32>(wshape_size);
    auto wsp = wshape.begin();
    for (uint32_t idx = 0; idx < wshape_size; idx++)
    {
      _inpsize->at<loco::DataType::S32>(idx) = int32_t(*wsp++);
    }
    _inpsize->name("inpsize");

    // trconv filter
    _filter = g->nodes()->create<luci::CircleConst>();
    _filter->dtype(loco::DataType::FLOAT32);
    _filter->shape(wshape);
    _filter->size<loco::DataType::FLOAT32>(elements_num);
    for (uint32_t idx = 0; idx < elements_num; idx++)
    {
      _filter->at<loco::DataType::FLOAT32>(idx) = float(idx);
    }
    _filter->name("filter");

    // trconv
    _tc = g->nodes()->create<luci::CircleTransposeConv>();
    _tc->dtype(loco::DataType::FLOAT32);
    _tc->name("trconv");
  }

protected:
  luci::CircleTransposeConv *_tc = nullptr;
  luci::CircleConst *_filter = nullptr;
  luci::CircleConst *_inpsize = nullptr;
};

class TrConvGraph : public TestIGraphlet, public TestOGraphlet, public TrConvGraphlet
{
public:
  TrConvGraph() = default;

  void init(const ShapeU32 shape)
  {
    TestIGraphlet::init(g(), shape);
    TestOGraphlet::init(g(), shape);
    TrConvGraphlet::init(g(), shape);

    // connect graph
    _tc->inputSizes(_inpsize);
    _tc->filter(_filter);
    _tc->outBackprop(input());

    output()->from(_tc);
  }
};

} // namespace

TEST(XpSepActFromTransposeConvPassTest, name)
{
  luci::XpSepActFromTransposeConvPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(XpSepActFromTransposeConvPassTest, separation_ok)
{
  TrConvGraph g;

  g.init({1, 4, 4, 3});

  auto tc_node = luci::test::first_node<luci::CircleTransposeConv>(g.g());
  ASSERT_NE(tc_node, nullptr);
  tc_node->fusedActivationFunction(luci::FusedActFunc::RELU);

  luci::XpSepActFromTransposeConvPass pass;
  EXPECT_EQ(pass.run(g.g()), true);

  auto la_node = dynamic_cast<luci::CircleRelu *>(g.output()->from());
  ASSERT_NE(la_node, nullptr);
  auto la_tc_node = dynamic_cast<luci::CircleTransposeConv *>(la_node->features());
  ASSERT_NE(la_tc_node, nullptr);
  ASSERT_EQ(la_tc_node->fusedActivationFunction(), luci::FusedActFunc::NONE);
}

TEST(XpSepActFromTransposeConvPassTest, none_act_NEG)
{
  TrConvGraph g;

  g.init({1, 4, 4, 3});

  auto tc_node = luci::test::first_node<luci::CircleTransposeConv>(g.g());
  ASSERT_NE(tc_node, nullptr);
  tc_node->fusedActivationFunction(luci::FusedActFunc::NONE);

  luci::XpSepActFromTransposeConvPass pass;
  EXPECT_NE(pass.run(g.g()), true);
}

TEST(XpSepActFromTransposeConvPassTest, invalid_act_NEG)
{
  TrConvGraph g;

  g.init({1, 4, 4, 3});

  auto tc_node = luci::test::first_node<luci::CircleTransposeConv>(g.g());
  ASSERT_NE(tc_node, nullptr);
  // leave activation as undefined

  luci::XpSepActFromTransposeConvPass pass;
  EXPECT_ANY_THROW(pass.run(g.g()));
}

TEST(XpSepActFromTransposeConvPassTest, invalid_dtype_NEG)
{
  TrConvGraph g;

  g.init({1, 4, 4, 3});

  auto tc_node = luci::test::first_node<luci::CircleTransposeConv>(g.g());
  ASSERT_NE(tc_node, nullptr);
  tc_node->dtype(loco::DataType::S16);

  luci::XpSepActFromTransposeConvPass pass;
  EXPECT_NE(pass.run(g.g()), true);
}
