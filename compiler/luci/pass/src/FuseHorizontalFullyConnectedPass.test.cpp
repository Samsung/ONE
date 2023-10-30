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

#include "luci/Pass/FuseHorizontalFullyConnectedPass.h"
#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

/*
 *  Before
 *
 *     +---- [In] ----+
 *     |              |
 *     V              V
 *   fc1 (w1, b1)   fc2 (w2, b2)
 *     |              |
 *     |              |
 *     +---> add <----+
 *            |
 *            V
 *          [Out]
 *
 *  After
 *
 *     [In]
 *      |
 *      V
 *     fc3 (w1+w2, b1+b2)
 *      |
 *      V
 *     [Out]
 */
class FuseHorizontalFCLayersTestGraph : public TestIOGraph
{
public:
  FuseHorizontalFCLayersTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 10}, {1, 10});

    _left_fc = g()->nodes()->create<luci::CircleFullyConnected>();
    _right_fc = g()->nodes()->create<luci::CircleFullyConnected>();
    _left_weight = g()->nodes()->create<luci::CircleConst>();
    _right_weight = g()->nodes()->create<luci::CircleConst>();

    _left_fc->name("left FC");
    _right_fc->name("right FC");
    _left_weight->name("left weight");
    _right_weight->name("right weight");

    _left_fc->dtype(loco::DataType::FLOAT32);
    _right_fc->dtype(loco::DataType::FLOAT32);

    _left_fc->shape_status(luci::ShapeStatus::VALID);
    _right_fc->shape_status(luci::ShapeStatus::VALID);

    _left_fc->fusedActivationFunction(luci::FusedActFunc::NONE);
    _right_fc->fusedActivationFunction(luci::FusedActFunc::NONE);

    _left_fc->rank(2);
    _right_fc->rank(2);

    _right_fc->dim(0) = 1;
    _right_fc->dim(1) = 10;

    _left_fc->dim(0) = 1;
    _left_fc->dim(1) = 10;

    _left_weight->rank(2);
    _left_weight->dtype(loco::DataType::FLOAT32);
    _left_weight->size<loco::DataType::FLOAT32>(5 * 10);
    for (uint32_t i = 0; i < 5 * 10; ++i)
    {
      _left_weight->at<loco::DataType::FLOAT32>(0) = 1.0f;
    }
    _left_weight->dim(0) = 5;
    _left_weight->dim(1) = 10;
    _left_weight->shape_status(luci::ShapeStatus::VALID);

    _right_weight->rank(2);
    _right_weight->dtype(loco::DataType::FLOAT32);
    _right_weight->size<loco::DataType::FLOAT32>(5 * 10);
    for (uint32_t i = 0; i < 5 * 10; ++i)
    {
      _right_weight->at<loco::DataType::FLOAT32>(0) = 2.0f;
    }
    _right_weight->dim(0) = 5;
    _right_weight->dim(1) = 10;
    _right_weight->shape_status(luci::ShapeStatus::VALID);

    const auto left_output_exclude = g()->nodes()->create<luci::CircleOutputExclude>();
    const auto right_output_exclude = g()->nodes()->create<luci::CircleOutputExclude>();

    _left_fc->input(input());
    _left_fc->weights(_left_weight);
    _left_fc->bias(left_output_exclude);
    _right_fc->input(input());
    _right_fc->weights(_right_weight);
    _right_fc->bias(right_output_exclude);

    _add = g()->nodes()->create<luci::CircleAdd>();
    _add->dtype(loco::DataType::FLOAT32);
    _add->rank(2);
    _add->dim(0) = 1;
    _add->dim(1) = 5;
    _add->x(_left_fc);
    _add->y(_right_fc);
    _add->shape_status(luci::ShapeStatus::VALID);

    output()->from(_add);
  }

  luci::CircleFullyConnected *get_left_fc() { return _left_fc; }

private:
  luci::CircleFullyConnected *_left_fc = nullptr;
  luci::CircleConst *_left_weight = nullptr;
  luci::CircleFullyConnected *_right_fc = nullptr;
  luci::CircleConst *_right_weight = nullptr;
  luci::CircleAdd *_add = nullptr;
};

} // namespace

TEST(FuseHorizontalFCLayersPassTest, name)
{
  luci::FuseHorizontalFullyConnectedPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseHorizontalFCLayersPassTest, fuse_horizontal_nodes)
{
  FuseHorizontalFCLayersTestGraph g;
  luci::FuseHorizontalFullyConnectedPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseHorizontalFCLayersPassTest, fuse_horizontal_nodes_NEG)
{
  FuseHorizontalFCLayersTestGraph g;
  luci::FuseHorizontalFullyConnectedPass pass;

  g.init();

  g.get_left_fc()->fusedActivationFunction(luci::FusedActFunc::RELU6);

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(FuseHorizontalFCLayersPassTest, wrong_dtype_NEG)
{
  FuseHorizontalFCLayersTestGraph g;
  luci::FuseHorizontalFullyConnectedPass pass;

  g.init();

  g.get_left_fc()->dtype(loco::DataType::S32);

  EXPECT_FALSE(pass.run(g.g()));
}
