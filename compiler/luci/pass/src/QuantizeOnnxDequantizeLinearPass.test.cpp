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

#include "QuantizeOnnxDequantizeLinearPass.h"
#include "PassTestGraphs.h"

#include <flatbuffers/flexbuffers.h>

#include <gtest/gtest.h>

namespace
{

template <loco::DataType DT>
class QuantizeOnnxDequantizeLinearTest : public luci::ConstantFoldingAddTestGraph,
                                         public ::testing::Test
{
public:
  QuantizeOnnxDequantizeLinearTest() : luci::ConstantFoldingAddTestGraph({2, 2, 2}, DT) {}

  virtual void SetUp() { init(); }

  loco::Node *createFoldedPattern() override
  {
    _dequantize = _g.nodes()->template create<luci::CircleCustom>(3, 1);
    _dequantize_out = _g.nodes()->template create<luci::CircleCustomOut>();
    _input = _g.nodes()->template create<luci::CircleConst>();
    _scale = _g.nodes()->template create<luci::CircleConst>();
    _zerop = _g.nodes()->template create<luci::CircleConst>();

    _dequantize->dtype(loco::DataType::FLOAT32);
    _dequantize_out->dtype(loco::DataType::FLOAT32);
    _input->dtype(DT);
    _scale->dtype(loco::DataType::FLOAT32);
    _zerop->dtype(DT);

    _input->shape({2, 2, 2});
    _scale->shape({2});
    _zerop->shape({2});

    _input->size<DT>(8);

    _scale->size<loco::DataType::FLOAT32>(2);
    _scale->at<loco::DataType::FLOAT32>(0) = 5.0;
    _scale->at<loco::DataType::FLOAT32>(1) = 10.0;

    _zerop->size<DT>(2);

    // custom option
    auto flex_buffers = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_buffers->StartMap();
    flex_buffers->Int("axis", 1);
    flex_buffers->EndMap(map_start);
    flex_buffers->Finish();

    _dequantize->inputs(0, _input);
    _dequantize->inputs(1, _scale);
    _dequantize->inputs(2, _zerop);
    _dequantize->custom_code("ONNXDequantizeLinear");
    _dequantize->custom_options(flex_buffers->GetBuffer());

    _dequantize_out->input(_dequantize);
    _dequantize_out->index(0);

    _input->name("input");
    _dequantize->name("dequantize");
    _dequantize_out->name("dequantize_out");

    return _dequantize_out;
  }

  void createNotQuantizablePattern() { _input->dtype(loco::DataType::FLOAT32); }

protected:
  luci::CircleCustom *_dequantize = nullptr;
  luci::CircleCustomOut *_dequantize_out = nullptr;
  luci::CircleConst *_input = nullptr;
  luci::CircleConst *_scale = nullptr;
  luci::CircleConst *_zerop = nullptr;
};

class S4QuantizeOnnxDequantizeLinearTest
  : public QuantizeOnnxDequantizeLinearTest<loco::DataType::U8>
{
  virtual void SetUp() override
  {
    init();

    // Input range [0, 15]
    for (uint32_t i = 0; i < _input->size<loco::DataType::U8>(); i++)
    {
      _input->at<loco::DataType::U8>(i) = 1;
    }

    // Zerop = 8
    for (uint32_t i = 0; i < _zerop->size<loco::DataType::U8>(); i++)
    {
      _zerop->at<loco::DataType::U8>(i) = 8;
    }
  }
};

class U4QuantizeOnnxDequantizeLinearTest
  : public QuantizeOnnxDequantizeLinearTest<loco::DataType::U8>
{
  virtual void SetUp() override
  {
    init();

    // Input range [0, 15]
    for (uint32_t i = 0; i < _input->size<loco::DataType::U8>(); i++)
    {
      _input->at<loco::DataType::U8>(i) = 1;
    }

    // Zerop = [0, 15]
    for (uint32_t i = 0; i < _zerop->size<loco::DataType::U8>(); i++)
    {
      _zerop->at<loco::DataType::U8>(i) = 1;
    }
  }
};

class U8QuantizeOnnxDequantizeLinearTest
  : public QuantizeOnnxDequantizeLinearTest<loco::DataType::U8>
{
  virtual void SetUp() override
  {
    init();

    // Input range [0, 255]
    for (uint32_t i = 0; i < _input->size<loco::DataType::U8>(); i++)
    {
      _input->at<loco::DataType::U8>(i) = 255;
    }

    // Zerop = [0, 255]
    for (uint32_t i = 0; i < _zerop->size<loco::DataType::U8>(); i++)
    {
      _zerop->at<loco::DataType::U8>(i) = 128;
    }
  }
};

class S16QuantizeOnnxDequantizeLinearTest
  : public QuantizeOnnxDequantizeLinearTest<loco::DataType::S16>
{
  virtual void SetUp() override
  {
    init();

    for (uint32_t i = 0; i < _input->size<loco::DataType::S16>(); i++)
    {
      _input->at<loco::DataType::S16>(i) = 1024;
    }

    for (uint32_t i = 0; i < _zerop->size<loco::DataType::S16>(); i++)
    {
      _zerop->at<loco::DataType::S16>(i) = 0;
    }
  }
};

} // namespace

TEST_F(S4QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic)
{
  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  EXPECT_EQ(loco::DataType::S4, folded_const->dtype());
}

TEST_F(S4QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic_NEG)
{
  createNotQuantizablePattern();

  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(U4QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic)
{
  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  EXPECT_EQ(loco::DataType::U4, folded_const->dtype());
}

TEST_F(U4QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic_NEG)
{
  createNotQuantizablePattern();

  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(U8QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic)
{
  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  EXPECT_EQ(loco::DataType::U8, folded_const->dtype());
}

TEST_F(U8QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic_NEG)
{
  createNotQuantizablePattern();

  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}

TEST_F(S16QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic)
{
  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_NE(nullptr, folded_const);

  EXPECT_EQ(loco::DataType::S16, folded_const->dtype());
}

TEST_F(S16QuantizeOnnxDequantizeLinearTest, quantize_onnx_dq_linear_basic_NEG)
{
  createNotQuantizablePattern();

  luci::QuantizeOnnxDequantizeLinearPass pass;
  while (pass.run(graph()))
    ;

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(nullptr, folded_const);
}
