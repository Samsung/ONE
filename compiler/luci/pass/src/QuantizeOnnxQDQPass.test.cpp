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

#include "QuantizeOnnxQDQPass.h"
#include "PassTestGraphs.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class U8OnnxQDQGraphlet
{
public:
  U8OnnxQDQGraphlet() = default;

  void init(loco::Graph *g)
  {
    _quantize = g->nodes()->create<luci::CircleCustom>(3, 1);
    _quantize_out = g->nodes()->create<luci::CircleCustomOut>();
    _dequantize = g->nodes()->create<luci::CircleCustom>(3, 1);
    _dequantize_out = g->nodes()->create<luci::CircleCustomOut>();
    _scale = g->nodes()->create<luci::CircleConst>();
    _zerop = g->nodes()->create<luci::CircleConst>();

    _quantize->dtype(loco::DataType::U8);
    _quantize_out->dtype(loco::DataType::U8);
    _dequantize->dtype(loco::DataType::FLOAT32);
    _dequantize_out->dtype(loco::DataType::FLOAT32);
    _scale->dtype(loco::DataType::FLOAT32);
    _zerop->dtype(loco::DataType::U8);

    _scale->shape({1});
    _zerop->shape({1});

    _scale->size<loco::DataType::FLOAT32>(1);
    _scale->at<loco::DataType::FLOAT32>(0) = 5.0;

    _zerop->size<loco::DataType::U8>(1);
    _zerop->at<loco::DataType::U8>(0) = 0;

    _quantize->custom_code("ONNXQuantizeLinear");
    _quantize_out->index(0);

    _dequantize->custom_code("ONNXDequantizeLinear");
    _dequantize_out->index(0);

    _scale->name("scale");
    _zerop->name("zerop");
    _quantize->name("quantize");
    _quantize_out->name("quantize_out");
    _dequantize->name("dequantize");
    _dequantize_out->name("dequantize_out");
  }

protected:
  luci::CircleCustom *_quantize = nullptr;
  luci::CircleCustomOut *_quantize_out = nullptr;
  luci::CircleCustom *_dequantize = nullptr;
  luci::CircleCustomOut *_dequantize_out = nullptr;
  luci::CircleConst *_scale = nullptr;
  luci::CircleConst *_zerop = nullptr;
};

class U8QuantizeOnnxQDQTestGraph : public TestIOGraph, public U8OnnxQDQGraphlet
{
public:
  void init(void)
  {
    TestIOGraph::init({2, 2, 2}, {2, 2, 2});
    U8OnnxQDQGraphlet::init(g());

    _quantize->inputs(0, input());
    _quantize->inputs(1, _scale);
    _quantize->inputs(2, _zerop);
    _quantize_out->input(_quantize);
    _dequantize->inputs(0, _quantize_out);
    _dequantize->inputs(1, _scale);
    _dequantize->inputs(2, _zerop);
    _dequantize_out->input(_dequantize);

    output()->from(_dequantize_out);
  }
};

class S16OnnxQDQGraphlet
{
public:
  S16OnnxQDQGraphlet() = default;

  void init(loco::Graph *g)
  {
    _quantize = g->nodes()->create<luci::CircleCustom>(3, 1);
    _quantize_out = g->nodes()->create<luci::CircleCustomOut>();
    _dequantize = g->nodes()->create<luci::CircleCustom>(3, 1);
    _dequantize_out = g->nodes()->create<luci::CircleCustomOut>();
    _scale = g->nodes()->create<luci::CircleConst>();
    _zerop = g->nodes()->create<luci::CircleConst>();

    _quantize->dtype(loco::DataType::S16);
    _quantize_out->dtype(loco::DataType::S16);
    _dequantize->dtype(loco::DataType::FLOAT32);
    _dequantize_out->dtype(loco::DataType::FLOAT32);
    _scale->dtype(loco::DataType::FLOAT32);
    _zerop->dtype(loco::DataType::S16);

    _scale->shape({1});
    _zerop->shape({1});

    _scale->size<loco::DataType::FLOAT32>(1);
    _scale->at<loco::DataType::FLOAT32>(0) = 5.0;

    _zerop->size<loco::DataType::S16>(1);
    _zerop->at<loco::DataType::S16>(0) = 0;

    _quantize->custom_code("ONNXQuantizeLinear");
    _quantize_out->index(0);

    _dequantize->custom_code("ONNXDequantizeLinear");
    _dequantize_out->index(0);

    _scale->name("scale");
    _zerop->name("zerop");
    _quantize->name("quantize");
    _quantize_out->name("quantize_out");
    _dequantize->name("dequantize");
    _dequantize_out->name("dequantize_out");
  }

protected:
  luci::CircleCustom *_quantize = nullptr;
  luci::CircleCustomOut *_quantize_out = nullptr;
  luci::CircleCustom *_dequantize = nullptr;
  luci::CircleCustomOut *_dequantize_out = nullptr;
  luci::CircleConst *_scale = nullptr;
  luci::CircleConst *_zerop = nullptr;
};

class S16QuantizeOnnxQDQTestGraph : public TestIOGraph, public S16OnnxQDQGraphlet
{
public:
  void init(void)
  {
    TestIOGraph::init({2, 2, 2}, {2, 2, 2});
    S16OnnxQDQGraphlet::init(g());

    _quantize->inputs(0, input());
    _quantize->inputs(1, _scale);
    _quantize->inputs(2, _zerop);
    _quantize_out->input(_quantize);
    _dequantize->inputs(0, _quantize_out);
    _dequantize->inputs(1, _scale);
    _dequantize->inputs(2, _zerop);
    _dequantize_out->input(_dequantize);

    output()->from(_dequantize_out);
  }
};

class S16ConstOnnxQDQTest : public luci::ConstantFoldingAddTestGraph, public ::testing::Test
{
public:
  S16ConstOnnxQDQTest() : luci::ConstantFoldingAddTestGraph({2, 2, 2}, loco::DataType::S16) {}

  virtual void SetUp() { init(); }

  loco::Node *createFoldedPattern() override
  {
    _quantize = _g.nodes()->create<luci::CircleCustom>(3, 1);
    _quantize_out = _g.nodes()->create<luci::CircleCustomOut>();
    _dequantize = _g.nodes()->create<luci::CircleCustom>(3, 1);
    _dequantize_out = _g.nodes()->create<luci::CircleCustomOut>();
    _scale = _g.nodes()->create<luci::CircleConst>();
    _zerop = _g.nodes()->create<luci::CircleConst>();
    _input = _g.nodes()->create<luci::CircleConst>();

    _quantize->dtype(loco::DataType::S16);
    _quantize_out->dtype(loco::DataType::S16);
    _dequantize->dtype(loco::DataType::FLOAT32);
    _dequantize_out->dtype(loco::DataType::FLOAT32);
    _scale->dtype(loco::DataType::FLOAT32);
    _zerop->dtype(loco::DataType::S16);
    _input->dtype(loco::DataType::FLOAT32);

    _scale->shape({1});
    _zerop->shape({1});
    _input->shape({2, 2, 2});

    _scale->size<loco::DataType::FLOAT32>(1);
    _scale->at<loco::DataType::FLOAT32>(0) = 5.0;

    _zerop->size<loco::DataType::S16>(1);
    _zerop->at<loco::DataType::S16>(0) = 0;

    _input->size<loco::DataType::FLOAT32>(8);
    for (uint32_t i = 0; i < 8; i++)
      _input->at<loco::DataType::FLOAT32>(i) = i;

    _quantize->custom_code("ONNXQuantizeLinear");
    _quantize_out->index(0);

    _dequantize->custom_code("ONNXDequantizeLinear");
    _dequantize_out->index(0);

    _scale->name("scale");
    _zerop->name("zerop");
    _quantize->name("quantize");
    _quantize_out->name("quantize_out");
    _dequantize->name("dequantize");
    _dequantize_out->name("dequantize_out");

    _quantize->inputs(0, _input);
    _quantize->inputs(1, _scale);
    _quantize->inputs(2, _zerop);
    _quantize_out->input(_quantize);
    _dequantize->inputs(0, _quantize_out);
    _dequantize->inputs(1, _scale);
    _dequantize->inputs(2, _zerop);
    _dequantize_out->input(_dequantize);

    return _dequantize_out;
  }

protected:
  luci::CircleCustom *_quantize = nullptr;
  luci::CircleCustomOut *_quantize_out = nullptr;
  luci::CircleCustom *_dequantize = nullptr;
  luci::CircleCustomOut *_dequantize_out = nullptr;
  luci::CircleConst *_scale = nullptr;
  luci::CircleConst *_zerop = nullptr;
  luci::CircleConst *_input = nullptr;
};

} // namespace

TEST(QuantizeOnnxQDQTest, s16_quantize_onnx_qdq)
{
  S16QuantizeOnnxQDQTestGraph g;

  luci::QuantizeOnnxQDQPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(QuantizeOnnxQDQTest, s16_quantize_onnx_qdq_NEG)
{
  S16QuantizeOnnxQDQTestGraph g;

  luci::QuantizeOnnxQDQPass pass;

  g.init();

  g.input()->dtype(loco::DataType::S16);

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(QuantizeOnnxQDQTest, u8_quantize_onnx_qdq)
{
  U8QuantizeOnnxQDQTestGraph g;

  luci::QuantizeOnnxQDQPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(QuantizeOnnxQDQTest, u8_quantize_onnx_qdq_NEG)
{
  U8QuantizeOnnxQDQTestGraph g;

  luci::QuantizeOnnxQDQPass pass;

  g.init();

  g.input()->dtype(loco::DataType::U8);

  EXPECT_FALSE(pass.run(g.g()));
}

TEST_F(S16ConstOnnxQDQTest, s16_const_qdq)
{
  luci::QuantizeOnnxQDQPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  // Chec type, shape, values of folded const
  EXPECT_EQ(loco::DataType::S16, folded_const->dtype());
  EXPECT_EQ(3, folded_const->rank());
  EXPECT_EQ(2, folded_const->dim(0).value());
  EXPECT_EQ(2, folded_const->dim(1).value());
  EXPECT_EQ(2, folded_const->dim(2).value());
  EXPECT_EQ(0, folded_const->at<loco::DataType::S16>(0));
  EXPECT_EQ(0, folded_const->at<loco::DataType::S16>(1));
  EXPECT_EQ(0, folded_const->at<loco::DataType::S16>(2));
  EXPECT_EQ(1, folded_const->at<loco::DataType::S16>(3));
  EXPECT_EQ(1, folded_const->at<loco::DataType::S16>(4));
  EXPECT_EQ(1, folded_const->at<loco::DataType::S16>(5));
  EXPECT_EQ(1, folded_const->at<loco::DataType::S16>(6));
  EXPECT_EQ(1, folded_const->at<loco::DataType::S16>(7));
}
