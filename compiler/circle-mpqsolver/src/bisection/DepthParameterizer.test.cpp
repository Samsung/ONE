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

#include <gtest/gtest.h>

#include "DepthParameterizer.h"
#include "core/TestHelper.h"

#include <luci/IR/CircleNodes.h>

namespace
{

class NConvGraph final : public mpqsolver::test::models::SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    _filter = _g->nodes()->create<luci::CircleConst>();
    _filter->dtype(loco::DataType::FLOAT32);
    _filter->shape({_channel_size, 1, 1, _channel_size});
    _filter->name("conv_filter");

    _bias = _g->nodes()->create<luci::CircleConst>();
    _bias->dtype(loco::DataType::FLOAT32);
    _bias->shape({_channel_size});
    _bias->name("conv_bias");

    _conv = _g->nodes()->create<luci::CircleConv2D>();
    _conv->padding(luci::Padding::SAME);
    _conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _conv->dtype(loco::DataType::FLOAT32);
    _conv->shape({1, _height, _width, _channel_size});
    _conv->name("conv");
    _conv->filter(_filter);
    _conv->bias(_bias);
    _conv->input(input);

    return _conv;
  }

public:
  luci::CircleConv2D *_conv = nullptr;
  luci::CircleConst *_filter = nullptr;
  luci::CircleConst *_bias = nullptr;
};

} // namespace

TEST(CircleMPQSolverDepthParameteriserTest, verifyResultsTest)
{
  auto m = luci::make_module();
  NConvGraph g;
  g.init();
  auto conv = g._conv;
  auto input = g._input;
  auto output = g._output;

  g.transfer_to(m.get());

  mpqsolver::bisection::NodeDepthType nodes_depth;
  float min_depth = std::numeric_limits<float>().max();
  float max_depth = -std::numeric_limits<float>().max();
  auto status = mpqsolver::bisection::compute_depth(m.get(), nodes_depth, min_depth, max_depth);

  EXPECT_TRUE(status == mpqsolver::bisection::ParameterizerResult::SUCCESS);
  EXPECT_TRUE(max_depth == 2 && min_depth == 0);
  EXPECT_TRUE(nodes_depth[input] == min_depth);
  EXPECT_TRUE(nodes_depth[conv] == 1);
  EXPECT_TRUE(nodes_depth[output] == max_depth);
}

TEST(CircleMPQSolverDepthParameteriserTest, verifyResultsTest_NEG)
{
  auto m = luci::make_module();
  mpqsolver::bisection::NodeDepthType nodes_depth;
  float min_depth = std::numeric_limits<float>().max();
  float max_depth = -std::numeric_limits<float>().max();
  auto status = mpqsolver::bisection::compute_depth(m.get(), nodes_depth, min_depth, max_depth);

  EXPECT_TRUE(status == mpqsolver::bisection::ParameterizerResult::FAILURE);
}
