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

#include "QuantizationUtils.h"

#include <luci/IR/CircleQuantParam.h>

#include <vector>

#include <gtest/gtest.h>

namespace
{

void addQuantParam(luci::CircleNode &node, const std::vector<float> &scale,
                   const std::vector<int64_t> &zp)
{
  assert(node.quantparam() == nullptr);

  auto quantparam = std::make_unique<luci::CircleQuantParam>();
  quantparam->scale = scale;
  quantparam->zerop = zp;
  node.quantparam(std::move(quantparam));
}

class SimpleConcatGraph
{
public:
  SimpleConcatGraph()
  {
    concat_node.dtype(loco::DataType::U8);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(loco::DataType::U8);
    input_2.dtype(loco::DataType::U8);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    addQuantParam(concat_node, {3.14}, {77});
    addQuantParam(input_1, {1.0}, {1});
    addQuantParam(input_2, {2.0}, {2});
  }

  ~SimpleConcatGraph()
  {
    concat_node.values(0, nullptr);
    concat_node.values(1, nullptr);
  }

public:
  luci::CircleConcatenation concat_node{2};
  luci::CircleConv2D input_1;
  luci::CircleConv2D input_2;
};

class SubsequentConcatGraph
{
public:
  SubsequentConcatGraph()
  {
    concat_node.dtype(loco::DataType::U8);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(loco::DataType::U8);
    input_2.dtype(loco::DataType::U8);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    addQuantParam(concat_node, {3.14}, {77});
    addQuantParam(input_1, {1.0}, {1});
    addQuantParam(input_2, {2.0}, {2});
  }

  ~SubsequentConcatGraph()
  {
    concat_node.values(0, nullptr);
    concat_node.values(1, nullptr);
  }

public:
  luci::CircleConcatenation concat_node{2};
  luci::CircleConcatenation input_1{2};
  luci::CircleConv2D input_2;
};

class ConstInputConcatGraph
{
public:
  ConstInputConcatGraph()
  {
    concat_node.dtype(loco::DataType::U8);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(loco::DataType::FLOAT32);
    input_1.size<loco::DataType::FLOAT32>(5);
    for (int i = 0; i < 5; i++)
    {
      // Set data {-2, -1, 0, 1, 2}
      input_1.at<loco::DataType::FLOAT32>(i) = i - 2.0;
    }

    input_2.dtype(loco::DataType::U8);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    addQuantParam(concat_node, {0.1}, {10});
    addQuantParam(input_1, {1.0}, {1});
    addQuantParam(input_2, {2.0}, {2});
  }

  ~ConstInputConcatGraph()
  {
    concat_node.values(0, nullptr);
    concat_node.values(1, nullptr);
  }

public:
  luci::CircleConcatenation concat_node{2};
  luci::CircleConst input_1;
  luci::CircleConv2D input_2;
};

} // namespace

TEST(PropagateConcatenationQparam, propagate_concat_quantparam)
{
  // Check cases where qparam of concat_node is propagated
  // (1) normal case: qparam is propagated to input_1 and input_2
  // (2) input used by other Op: input_1 is an input of input_2. qparam is propagated only to
  // input_2
  // (3) subsequent concat: input_1 is concat. qparam is propagated only to input_2
  // (4) const input: input_1 is const. constant values are quantized

  // normal case: qparam of concat_node is propagated to input_1 and input_2
  SimpleConcatGraph g;
  luci::propagate_concat_quantparam(&g.concat_node);
  EXPECT_FLOAT_EQ(g.concat_node.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.concat_node.quantparam()->zerop[0], 77);
  EXPECT_FLOAT_EQ(g.input_1.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.input_1.quantparam()->zerop[0], 77);
  EXPECT_FLOAT_EQ(g.input_2.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.input_2.quantparam()->zerop[0], 77);

  // input_1 is an input of input_2. qparam is propagated only to input_2
  SimpleConcatGraph g2;
  g2.input_2.input(&g2.input_1);
  luci::propagate_concat_quantparam(&g2.concat_node);
  EXPECT_FLOAT_EQ(g2.concat_node.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g2.concat_node.quantparam()->zerop[0], 77);
  EXPECT_FLOAT_EQ(g2.input_1.quantparam()->scale[0], 1.0);
  EXPECT_EQ(g2.input_1.quantparam()->zerop[0], 1);
  EXPECT_FLOAT_EQ(g2.input_2.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g2.input_2.quantparam()->zerop[0], 77);

  // input_1 is concat. qparam is propagated only to input_2
  SubsequentConcatGraph sg;
  luci::propagate_concat_quantparam(&sg.concat_node);
  EXPECT_FLOAT_EQ(sg.concat_node.quantparam()->scale[0], 3.14);
  EXPECT_EQ(sg.concat_node.quantparam()->zerop[0], 77);
  EXPECT_FLOAT_EQ(sg.input_1.quantparam()->scale[0], 1.0);
  EXPECT_EQ(sg.input_1.quantparam()->zerop[0], 1);
  EXPECT_FLOAT_EQ(sg.input_2.quantparam()->scale[0], 3.14);
  EXPECT_EQ(sg.input_2.quantparam()->zerop[0], 77);

  // input_1 is const. const values are quantized with the qparam of concat
  ConstInputConcatGraph cg;
  luci::propagate_concat_quantparam(&cg.concat_node);
  EXPECT_FLOAT_EQ(cg.concat_node.quantparam()->scale[0], 0.1);
  EXPECT_EQ(cg.concat_node.quantparam()->zerop[0], 10);
  EXPECT_FLOAT_EQ(cg.input_1.quantparam()->scale[0], 0.1);
  EXPECT_EQ(cg.input_1.quantparam()->zerop[0], 10);
  EXPECT_FLOAT_EQ(cg.input_2.quantparam()->scale[0], 0.1);
  EXPECT_EQ(cg.input_2.quantparam()->zerop[0], 10);
  EXPECT_EQ(cg.input_1.dtype(), loco::DataType::U8);
  EXPECT_EQ(cg.input_1.at<loco::DataType::U8>(0), 0);
  EXPECT_EQ(cg.input_1.at<loco::DataType::U8>(1), 0);
  EXPECT_EQ(cg.input_1.at<loco::DataType::U8>(2), 10);
  EXPECT_EQ(cg.input_1.at<loco::DataType::U8>(3), 20);
  EXPECT_EQ(cg.input_1.at<loco::DataType::U8>(4), 30);
}

TEST(PropagateConcatenationQparam, propagate_concat_quantparam_NEG)
{
  // Check negative cases where qparam is not propagated
  // (1) concat is not uint8-quantized
  // (2) concat has fused activation function

  SimpleConcatGraph g;

  // concat is not uint8-quantized
  g.concat_node.dtype(loco::DataType::S8);
  luci::propagate_concat_quantparam(&g.concat_node);
  EXPECT_FLOAT_EQ(g.concat_node.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.concat_node.quantparam()->zerop[0], 77);
  EXPECT_FLOAT_EQ(g.input_1.quantparam()->scale[0], 1.0);
  EXPECT_EQ(g.input_1.quantparam()->zerop[0], 1);
  EXPECT_FLOAT_EQ(g.input_2.quantparam()->scale[0], 2.0);
  EXPECT_EQ(g.input_2.quantparam()->zerop[0], 2);
  g.concat_node.dtype(loco::DataType::U8);

  // concat has fused activation function
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  luci::propagate_concat_quantparam(&g.concat_node);
  EXPECT_FLOAT_EQ(g.concat_node.quantparam()->scale[0], 3.14);
  EXPECT_EQ(g.concat_node.quantparam()->zerop[0], 77);
  EXPECT_FLOAT_EQ(g.input_1.quantparam()->scale[0], 1.0);
  EXPECT_EQ(g.input_1.quantparam()->zerop[0], 1);
  EXPECT_FLOAT_EQ(g.input_2.quantparam()->scale[0], 2.0);
  EXPECT_EQ(g.input_2.quantparam()->zerop[0], 2);
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
}
