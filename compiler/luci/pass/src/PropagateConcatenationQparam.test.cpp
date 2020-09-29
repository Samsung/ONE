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

#include <math.h>
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

int32_t quantize(float f, luci::CircleQuantParam *qparam)
{
  float scale = qparam->scale[0];
  int64_t zp = qparam->zerop[0];

  return std::round(f / scale) + zp;
}

class SimpleConcatGraph
{
public:
  SimpleConcatGraph(loco::DataType quant_type)
  {
    concat_node.dtype(quant_type);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(quant_type);
    input_2.dtype(quant_type);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    if (quant_type == loco::DataType::U8)
    {
      addQuantParam(concat_node, {3.14}, {77});
      addQuantParam(input_1, {1.0}, {1});
      addQuantParam(input_2, {2.0}, {2});
    }
    else if (quant_type == loco::DataType::S16)
    {
      addQuantParam(concat_node, {3.14}, {0});
      addQuantParam(input_1, {1.0}, {0});
      addQuantParam(input_2, {2.0}, {0});
    }
    else
    {
      throw std::runtime_error("Unsupported quantization type");
    }
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
  SubsequentConcatGraph(loco::DataType quant_type)
  {
    concat_node.dtype(quant_type);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(quant_type);
    input_2.dtype(quant_type);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    if (quant_type == loco::DataType::U8)
    {
      addQuantParam(concat_node, {3.14}, {77});
      addQuantParam(input_1, {1.0}, {1});
      addQuantParam(input_2, {2.0}, {2});
    }
    else if (quant_type == loco::DataType::S16)
    {
      addQuantParam(concat_node, {3.14}, {0});
      addQuantParam(input_1, {1.0}, {0});
      addQuantParam(input_2, {2.0}, {0});
    }
    else
    {
      throw std::runtime_error("Unsupported quantization type");
    }
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
  ConstInputConcatGraph(loco::DataType quant_type)
  {
    concat_node.dtype(quant_type);
    concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1.dtype(loco::DataType::FLOAT32);
    input_1.size<loco::DataType::FLOAT32>(5);
    for (int i = 0; i < 5; i++)
    {
      // Set data {-2, -1, 0, 1, 2}
      input_1.at<loco::DataType::FLOAT32>(i) = i - 2.0;
    }

    input_2.dtype(quant_type);

    concat_node.values(0, &input_1);
    concat_node.values(1, &input_2);

    if (quant_type == loco::DataType::U8)
    {
      addQuantParam(concat_node, {0.1}, {10});
      addQuantParam(input_2, {2.0}, {2});
    }
    else if (quant_type == loco::DataType::S16)
    {
      addQuantParam(concat_node, {0.1}, {0});
      addQuantParam(input_2, {2.0}, {0});
    }
    else
    {
      throw std::runtime_error("Unsupported quantization type");
    }
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

TEST(PropagateConcatenationQparam, propagate_concat_quantparam_u8)
{
  // Check cases where qparam of concat_node is propagated
  // (1) normal case: qparam is propagated to input_1 and input_2
  // (2) input used by other Op: input_1 is an input of input_2. qparam is propagated only to
  // input_2
  // (3) subsequent concat: input_1 is concat. qparam is propagated only to input_2
  // (4) const input: input_1 is const. constant values are quantized

  // normal case: qparam of concat_node is propagated to input_1 and input_2
  SimpleConcatGraph g(loco::DataType::U8);
  luci::propagate_concat_quantparam(&g.concat_node, loco::DataType::U8);
  EXPECT_FLOAT_EQ(3.14, g.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(77, g.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, g.input_1.quantparam()->scale[0]);
  EXPECT_EQ(77, g.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, g.input_2.quantparam()->scale[0]);
  EXPECT_EQ(77, g.input_2.quantparam()->zerop[0]);

  // input_1 is an input of input_2. qparam is propagated only to input_2
  SimpleConcatGraph g2(loco::DataType::U8);
  g2.input_2.input(&g2.input_1);
  luci::propagate_concat_quantparam(&g2.concat_node, loco::DataType::U8);
  EXPECT_FLOAT_EQ(3.14, g2.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(77, g2.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(1.0, g2.input_1.quantparam()->scale[0]);
  EXPECT_EQ(1, g2.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, g2.input_2.quantparam()->scale[0]);
  EXPECT_EQ(77, g2.input_2.quantparam()->zerop[0]);

  // input_1 is concat. qparam is propagated only to input_2
  SubsequentConcatGraph sg(loco::DataType::U8);
  luci::propagate_concat_quantparam(&sg.concat_node, loco::DataType::U8);
  EXPECT_FLOAT_EQ(3.14, sg.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(77, sg.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(1.0, sg.input_1.quantparam()->scale[0]);
  EXPECT_EQ(1, sg.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, sg.input_2.quantparam()->scale[0]);
  EXPECT_EQ(77, sg.input_2.quantparam()->zerop[0]);

  // input_1 is const. const values are quantized with the qparam of concat
  ConstInputConcatGraph cg(loco::DataType::U8);
  luci::propagate_concat_quantparam(&cg.concat_node, loco::DataType::U8);
  EXPECT_FLOAT_EQ(0.1, cg.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(10, cg.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(0.1, cg.input_1.quantparam()->scale[0]);
  EXPECT_EQ(10, cg.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(0.1, cg.input_2.quantparam()->scale[0]);
  EXPECT_EQ(10, cg.input_2.quantparam()->zerop[0]);
  EXPECT_EQ(loco::DataType::U8, cg.input_1.dtype());
  EXPECT_EQ(0, cg.input_1.at<loco::DataType::U8>(0));
  EXPECT_EQ(0, cg.input_1.at<loco::DataType::U8>(1));
  EXPECT_EQ(10, cg.input_1.at<loco::DataType::U8>(2));
  EXPECT_EQ(20, cg.input_1.at<loco::DataType::U8>(3));
  EXPECT_EQ(30, cg.input_1.at<loco::DataType::U8>(4));
}

TEST(PropagateConcatenationQparam, propagate_concat_quantparam_u8_NEG)
{
  // Check negative cases where qparam is not propagated
  // (1) concat has fused activation function
  // (2) concat has fused activation function and input is const

  SimpleConcatGraph g(loco::DataType::U8);

  // concat has fused activation function
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  luci::propagate_concat_quantparam(&g.concat_node, loco::DataType::U8);
  EXPECT_FLOAT_EQ(3.14, g.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(77, g.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(1.0, g.input_1.quantparam()->scale[0]);
  EXPECT_EQ(1, g.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(2.0, g.input_2.quantparam()->scale[0]);
  EXPECT_EQ(2, g.input_2.quantparam()->zerop[0]);
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);

  // concat has fused activation function and input_1 is const.
  // const values are quantized using its min/max
  ConstInputConcatGraph cg(loco::DataType::U8);
  cg.concat_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  luci::propagate_concat_quantparam(&cg.concat_node, loco::DataType::U8);
  EXPECT_FLOAT_EQ(0.1, cg.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(10, cg.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(0.015686275, cg.input_1.quantparam()->scale[0]);
  EXPECT_EQ(128, cg.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(2.0, cg.input_2.quantparam()->scale[0]);
  EXPECT_EQ(2, cg.input_2.quantparam()->zerop[0]);
  EXPECT_EQ(loco::DataType::U8, cg.input_1.dtype());
  EXPECT_EQ(quantize(-2, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::U8>(0));
  EXPECT_EQ(quantize(-1, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::U8>(1));
  EXPECT_EQ(quantize(0, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::U8>(2));
  EXPECT_EQ(quantize(1, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::U8>(3));
  EXPECT_EQ(quantize(2, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::U8>(4));
}

TEST(PropagateConcatenationQparam, propagate_concat_quantparam_i16)
{
  // Check cases where qparam of concat_node is propagated
  // (1) normal case: qparam is propagated to input_1 and input_2
  // (2) input used by other Op: input_1 is an input of input_2. qparam is propagated only to
  // input_2
  // (3) subsequent concat: input_1 is concat. qparam is propagated only to input_2
  // (4) const input: input_1 is const. constant values are quantized

  // normal case: qparam of concat_node is propagated to input_1 and input_2
  SimpleConcatGraph g(loco::DataType::S16);
  luci::propagate_concat_quantparam(&g.concat_node, loco::DataType::S16);
  EXPECT_FLOAT_EQ(3.14, g.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(0, g.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, g.input_1.quantparam()->scale[0]);
  EXPECT_EQ(0, g.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, g.input_2.quantparam()->scale[0]);
  EXPECT_EQ(0, g.input_2.quantparam()->zerop[0]);

  // input_1 is an input of input_2. qparam is propagated only to input_2
  SimpleConcatGraph g2(loco::DataType::S16);
  g2.input_2.input(&g2.input_1);
  luci::propagate_concat_quantparam(&g2.concat_node, loco::DataType::S16);
  EXPECT_FLOAT_EQ(3.14, g2.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(0, g2.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(1.0, g2.input_1.quantparam()->scale[0]);
  EXPECT_EQ(0, g2.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, g2.input_2.quantparam()->scale[0]);
  EXPECT_EQ(0, g2.input_2.quantparam()->zerop[0]);

  // input_1 is concat. qparam is propagated only to input_2
  SubsequentConcatGraph sg(loco::DataType::S16);
  luci::propagate_concat_quantparam(&sg.concat_node, loco::DataType::S16);
  EXPECT_FLOAT_EQ(3.14, sg.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(0, sg.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(1.0, sg.input_1.quantparam()->scale[0]);
  EXPECT_EQ(0, sg.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(3.14, sg.input_2.quantparam()->scale[0]);
  EXPECT_EQ(0, sg.input_2.quantparam()->zerop[0]);

  // input_1 is const. const values are quantized with the qparam of concat
  ConstInputConcatGraph cg(loco::DataType::S16);
  luci::propagate_concat_quantparam(&cg.concat_node, loco::DataType::S16);
  EXPECT_FLOAT_EQ(0.1, cg.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(0, cg.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(0.1, cg.input_1.quantparam()->scale[0]);
  EXPECT_EQ(0, cg.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(0.1, cg.input_2.quantparam()->scale[0]);
  EXPECT_EQ(0, cg.input_2.quantparam()->zerop[0]);
  EXPECT_EQ(loco::DataType::S16, cg.input_1.dtype());
  EXPECT_EQ(-20, cg.input_1.at<loco::DataType::S16>(0));
  EXPECT_EQ(-10, cg.input_1.at<loco::DataType::S16>(1));
  EXPECT_EQ(0, cg.input_1.at<loco::DataType::S16>(2));
  EXPECT_EQ(10, cg.input_1.at<loco::DataType::S16>(3));
  EXPECT_EQ(20, cg.input_1.at<loco::DataType::S16>(4));
}

TEST(PropagateConcatenationQparam, propagate_concat_quantparam_i16_NEG)
{
  // Check negative cases where qparam is not propagated
  // (1) concat has fused activation function
  // (2) concat has fused activation function and input is const

  SimpleConcatGraph g(loco::DataType::S16);

  // concat has fused activation function
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  luci::propagate_concat_quantparam(&g.concat_node, loco::DataType::S16);
  EXPECT_FLOAT_EQ(3.14, g.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(0, g.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(1.0, g.input_1.quantparam()->scale[0]);
  EXPECT_EQ(0, g.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(2.0, g.input_2.quantparam()->scale[0]);
  EXPECT_EQ(0, g.input_2.quantparam()->zerop[0]);
  g.concat_node.fusedActivationFunction(luci::FusedActFunc::NONE);

  // concat has fused activation function and input_1 is const.
  // const values are quantized using its min/max
  ConstInputConcatGraph cg(loco::DataType::S16);
  cg.concat_node.fusedActivationFunction(luci::FusedActFunc::RELU);
  luci::propagate_concat_quantparam(&cg.concat_node, loco::DataType::S16);
  EXPECT_FLOAT_EQ(0.1, cg.concat_node.quantparam()->scale[0]);
  EXPECT_EQ(0, cg.concat_node.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(0.000061037, cg.input_1.quantparam()->scale[0]);
  EXPECT_EQ(0, cg.input_1.quantparam()->zerop[0]);
  EXPECT_FLOAT_EQ(2.0, cg.input_2.quantparam()->scale[0]);
  EXPECT_EQ(0, cg.input_2.quantparam()->zerop[0]);
  EXPECT_EQ(loco::DataType::S16, cg.input_1.dtype());
  EXPECT_EQ(quantize(-2, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::S16>(0));
  EXPECT_EQ(quantize(-1, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::S16>(1));
  EXPECT_EQ(quantize(0, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::S16>(2));
  EXPECT_EQ(quantize(1, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::S16>(3));
  EXPECT_EQ(quantize(2, cg.input_1.quantparam()), cg.input_1.at<loco::DataType::S16>(4));
}
