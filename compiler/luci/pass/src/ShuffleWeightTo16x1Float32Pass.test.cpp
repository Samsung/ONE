/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ShuffleWeightTo16x1Float32Pass.h"

#include <luci/IR/CircleNodes.h>

#include "test/TestIOGraph.h"
#include "test/TestFirstNode.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class FCGraphlet
{
public:
  FCGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 wshape)
  {
    const uint32_t elements_num = num_elements(wshape);

    // fc weights
    _weights = g->nodes()->create<luci::CircleConst>();
    _weights->dtype(loco::DataType::FLOAT32);
    _weights->shape(wshape);
    _weights->size<loco::DataType::FLOAT32>(elements_num);
    for (uint32_t idx = 0; idx < elements_num; idx++)
    {
      _weights->at<loco::DataType::FLOAT32>(idx) = idx;
    }
    _weights->name("weights");

    // fc
    _fc = g->nodes()->create<luci::CircleFullyConnected>();
    _fc->dtype(loco::DataType::FLOAT32);
    _fc->name("fc");
  }

protected:
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleConst *_weights = nullptr;
};

class FCGraph : public TestIGraphlet, public TestOGraphlet, public FCGraphlet
{
public:
  FCGraph() = default;

  void init(const ShapeU32 shape, const ShapeU32 wshape)
  {
    TestIGraphlet::init(g(), shape);
    TestOGraphlet::init(g(), shape);
    FCGraphlet::init(g(), wshape);

    // connect graph
    _fc->input(input());
    _fc->weights(_weights);

    output()->from(_fc);
  }
};

} // namespace

TEST(ShuffleWeightTo16x1Float32PassTest, name)
{
  luci::ShuffleWeightTo16x1Float32Pass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

const uint32_t ROW = 16;
const uint32_t COL = 2;

TEST(ShuffleWeightTo16x1Float32PassTest, SimpleTest1)
{
  FCGraph g;

  g.init({ROW, COL}, {ROW, COL});

  auto fc_node = luci::test::first_node<luci::CircleFullyConnected>(g.g());
  ASSERT_NE(fc_node, nullptr);
  auto weights = loco::must_cast<luci::CircleConst *>(fc_node->weights());
  // before
  ASSERT_EQ(0, weights->at<loco::DataType::FLOAT32>(0));
  ASSERT_EQ(1, weights->at<loco::DataType::FLOAT32>(1));
  ASSERT_EQ(2, weights->at<loco::DataType::FLOAT32>(2));
  ASSERT_EQ(3, weights->at<loco::DataType::FLOAT32>(3));
  ASSERT_EQ(4, weights->at<loco::DataType::FLOAT32>(4));
  ASSERT_EQ(5, weights->at<loco::DataType::FLOAT32>(5));
  ASSERT_EQ(6, weights->at<loco::DataType::FLOAT32>(6));
  ASSERT_EQ(7, weights->at<loco::DataType::FLOAT32>(7));
  ASSERT_EQ(8, weights->at<loco::DataType::FLOAT32>(8));
  ASSERT_EQ(9, weights->at<loco::DataType::FLOAT32>(9));
  ASSERT_EQ(10, weights->at<loco::DataType::FLOAT32>(10));
  ASSERT_EQ(11, weights->at<loco::DataType::FLOAT32>(11));
  ASSERT_EQ(12, weights->at<loco::DataType::FLOAT32>(12));
  ASSERT_EQ(13, weights->at<loco::DataType::FLOAT32>(13));
  ASSERT_EQ(14, weights->at<loco::DataType::FLOAT32>(14));
  ASSERT_EQ(15, weights->at<loco::DataType::FLOAT32>(15));

  luci::ShuffleWeightTo16x1Float32Pass pass;
  while (pass.run(g.g()))
    ;

  weights = loco::must_cast<luci::CircleConst *>(fc_node->weights());
  // after
  ASSERT_EQ(0, weights->at<loco::DataType::FLOAT32>(0));
  ASSERT_EQ(2, weights->at<loco::DataType::FLOAT32>(1));
  ASSERT_EQ(4, weights->at<loco::DataType::FLOAT32>(2));
  ASSERT_EQ(6, weights->at<loco::DataType::FLOAT32>(3));
  ASSERT_EQ(8, weights->at<loco::DataType::FLOAT32>(4));
  ASSERT_EQ(10, weights->at<loco::DataType::FLOAT32>(5));
  ASSERT_EQ(12, weights->at<loco::DataType::FLOAT32>(6));
  ASSERT_EQ(14, weights->at<loco::DataType::FLOAT32>(7));
  ASSERT_EQ(16, weights->at<loco::DataType::FLOAT32>(8));
  ASSERT_EQ(18, weights->at<loco::DataType::FLOAT32>(9));
  ASSERT_EQ(20, weights->at<loco::DataType::FLOAT32>(10));
  ASSERT_EQ(22, weights->at<loco::DataType::FLOAT32>(11));
  ASSERT_EQ(24, weights->at<loco::DataType::FLOAT32>(12));
  ASSERT_EQ(26, weights->at<loco::DataType::FLOAT32>(13));
  ASSERT_EQ(28, weights->at<loco::DataType::FLOAT32>(14));
  ASSERT_EQ(30, weights->at<loco::DataType::FLOAT32>(15));
}

TEST(ShuffleWeightTo16x1Float32PassTest, invalid_weight_shape_NEG)
{
  FCGraph g;

  g.init({ROW, COL}, {1, ROW, COL, 1});

  auto fc_node = luci::test::first_node<luci::CircleFullyConnected>(g.g());
  ASSERT_NE(fc_node, nullptr);

  luci::ShuffleWeightTo16x1Float32Pass pass;
  auto ret = pass.run(g.g());

  ASSERT_FALSE(ret);
}

TEST(ShuffleWeightTo16x1Float32PassTest, invalid_weight_row16_NEG)
{
  FCGraph g;

  g.init({COL, ROW}, {COL, ROW});

  auto fc_node = luci::test::first_node<luci::CircleFullyConnected>(g.g());
  ASSERT_NE(fc_node, nullptr);

  luci::ShuffleWeightTo16x1Float32Pass pass;
  auto ret = pass.run(g.g());

  ASSERT_FALSE(ret);
}
