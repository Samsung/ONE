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

#include "luci/Pass/FoldDepthwiseConv2DPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <limits> // std::numeric_limits

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph has an DepthwiseConv2D Op with constant inputs
 *
 *    BEFORE
 *
 *    [CircleConst] [CircleConst]
 *               |   |
 *       [CircleDepthwiseConv2D]
 *
 *    AFTER
 *
 *           [CircleConst]
 */
class FoldDepthwiseConv2DTest : public luci::ConstantFoldingTestGraph, public ::testing::Test
{
public:
  FoldDepthwiseConv2DTest() : luci::ConstantFoldingTestGraph({1, 4, 4, 1}, loco::DataType::FLOAT32)
  {
    _dconv = _g.nodes()->create<luci::CircleDepthwiseConv2D>();
    _dconv_input = _g.nodes()->create<luci::CircleConst>();
    _dconv_filter = _g.nodes()->create<luci::CircleConst>();
    _dconv_bias = _g.nodes()->create<luci::CircleConst>();

    _dconv->dtype(loco::DataType::FLOAT32);
    _dconv->padding(luci::Padding::VALID);
    _dconv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _dconv->input(_dconv_input);
    _dconv->filter(_dconv_filter);
    _dconv->bias(_dconv_bias);
    _dconv->shape({1, 4, 4, 1});
    _dconv->shape_status(luci::ShapeStatus::VALID);
    _dconv->stride()->h(1);
    _dconv->stride()->w(1);
    _dconv->depthMultiplier(1);

    _dconv_input->dtype(loco::DataType::FLOAT32);
    _dconv_input->shape({1, 4, 4, 1});
    _dconv_input->size<loco::DataType::FLOAT32>(16);

    _dconv_filter->dtype(loco::DataType::FLOAT32);
    _dconv_filter->shape({1, 1, 1, 1});
    _dconv_filter->size<loco::DataType::FLOAT32>(1);

    _dconv_bias->dtype(loco::DataType::FLOAT32);
    _dconv_bias->shape({1});
    _dconv_bias->size<loco::DataType::FLOAT32>(1);

    _output->from(_dconv);
  }

protected:
  void init() final {}

protected:
  loco::Node *createFoldedPattern() final { return nullptr; }

protected:
  luci::CircleConst *getFoldedPattern() final
  {
    return loco::must_cast<luci::CircleConst *>(_output->from());
  }

protected:
  luci::CircleDepthwiseConv2D *_dconv = nullptr;
  luci::CircleConst *_dconv_input = nullptr;
  luci::CircleConst *_dconv_filter = nullptr;
  luci::CircleConst *_dconv_bias = nullptr;
};

} // namespace

TEST(FoldDepthwiseConv2DPass, name)
{
  luci::FoldDepthwiseConv2DPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldDepthwiseConv2DTest, fold_depthwise_conv2d)
{
  for (uint32_t i = 0; i < 16; ++i)
    _dconv_input->at<loco::DataType::FLOAT32>(i) = 0.5;
  _dconv_filter->at<loco::DataType::FLOAT32>(0) = 0.5;

  luci::FoldDepthwiseConv2DPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(folded_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_NEAR(folded_const->at<loco::DataType::FLOAT32>(0), 0.25,
              std::numeric_limits<float>::min());
  EXPECT_NEAR(folded_const->at<loco::DataType::FLOAT32>(15), 0.25,
              std::numeric_limits<float>::min());
}

TEST_F(FoldDepthwiseConv2DTest, fold_non_constant_NEG)
{
  _dconv->input(_input);

  luci::FoldDepthwiseConv2DPass pass;
  ASSERT_FALSE(pass.run(&_g));
}
