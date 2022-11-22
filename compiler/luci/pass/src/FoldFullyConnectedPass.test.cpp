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

#include "luci/Pass/FoldFullyConnectedPass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <limits> // std::numeric_limits

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph has an FullyConnected Op with constant inputs
 *
 *    BEFORE
 *
 *    [CircleConst] [CircleConst]
 *               |   |
 *       [CircleFullyConnected]
 *
 *    AFTER
 *
 *           [CircleConst]
 */
class FoldFullyConnectedTest : public luci::ConstantFoldingTestGraph, public ::testing::Test
{
public:
  FoldFullyConnectedTest() : luci::ConstantFoldingTestGraph({80}, loco::DataType::FLOAT32)
  {
    _fc = _g.nodes()->create<luci::CircleFullyConnected>();
    _fc_input = _g.nodes()->create<luci::CircleConst>();
    _fc_weights = _g.nodes()->create<luci::CircleConst>();
    _fc_bias = _g.nodes()->create<luci::CircleConst>();

    _fc->dtype(loco::DataType::FLOAT32);
    _fc->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc->input(_fc_input);
    _fc->weights(_fc_weights);
    _fc->bias(_fc_bias);
    _fc->shape({32});
    _fc->weights_format(luci::CircleFullyConnected::WeightsFormat::DEFAULT);
    _fc->keep_num_dims(true);

    _fc_input->dtype(loco::DataType::FLOAT32);
    _fc_input->shape({80});
    _fc_input->size<loco::DataType::FLOAT32>(80);

    _fc_weights->dtype(loco::DataType::FLOAT32);
    _fc_weights->shape({32, 80});
    _fc_weights->size<loco::DataType::FLOAT32>(32 * 80);

    _fc_bias->dtype(loco::DataType::FLOAT32);
    _fc_bias->shape({1, 32});
    _fc_bias->size<loco::DataType::FLOAT32>(32);

    for (uint32_t i = 0; i < 80; ++i)
      _fc_input->at<loco::DataType::FLOAT32>(i) = 1.0;

    for (uint32_t i = 0; i < 80 * 32; ++i)
      _fc_weights->at<loco::DataType::FLOAT32>(i) = 1.0;

    for (uint32_t i = 0; i < 32; ++i)
      _fc_bias->at<loco::DataType::FLOAT32>(i) = 0.0;

    _output->from(_fc);
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
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleConst *_fc_input = nullptr;
  luci::CircleConst *_fc_weights = nullptr;
  luci::CircleConst *_fc_bias = nullptr;
};

} // namespace

TEST_F(FoldFullyConnectedTest, fold_fc)
{
  luci::FoldFullyConnectedPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(folded_const->dtype(), loco::DataType::FLOAT32);
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(32, folded_const->dim(0));
  EXPECT_EQ(32, folded_const->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < 32; ++i)
    EXPECT_NEAR(folded_const->at<loco::DataType::FLOAT32>(i), 80,
                std::numeric_limits<float>::min());
}

TEST_F(FoldFullyConnectedTest, fold_fc_no_bias)
{
  auto no_bias = _g.nodes()->create<luci::CircleOutputExclude>();
  _fc->bias(no_bias);

  luci::FoldFullyConnectedPass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(loco::DataType::FLOAT32, folded_const->dtype());
  EXPECT_EQ(1, folded_const->rank());
  EXPECT_EQ(32, folded_const->dim(0));
  EXPECT_EQ(32, folded_const->size<loco::DataType::FLOAT32>());
  for (uint32_t i = 0; i < 32; ++i)
    EXPECT_NEAR(folded_const->at<loco::DataType::FLOAT32>(i), 80,
                std::numeric_limits<float>::min());
}

TEST_F(FoldFullyConnectedTest, fold_fc_NEG)
{
  auto new_fc = _g.nodes()->create<luci::CircleFullyConnected>();
  _fc->input(new_fc);

  luci::FoldFullyConnectedPass pass;
  ASSERT_FALSE(pass.run(&_g));
}
