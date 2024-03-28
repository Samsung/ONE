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

#include "luci/Pass/QuantizeOnnxFakeQuantModelPass.h"
#include "PassTestGraphs.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class S16OnnxFakeQuantGraphlet
{
public:
  S16OnnxFakeQuantGraphlet() = default;

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

class S16QuantizeOnnxFakeQuantModelTestGraph : public TestIOGraph, public S16OnnxFakeQuantGraphlet
{
public:
  void init(void)
  {
    TestIOGraph::init({2, 2, 2}, {2, 2, 2});
    S16OnnxFakeQuantGraphlet::init(g());

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

} // namespace

TEST(QuantizeOnnxFakeQuantModelTest, s16_quantize_onnx_qdq)
{
  S16QuantizeOnnxFakeQuantModelTestGraph g;

  auto ctx = std::make_unique<luci::QuantizeOnnxFakeQuantModelPass::Context>();
  {
    ctx->default_activation_dtype = loco::DataType::S16;
  }

  luci::QuantizeOnnxFakeQuantModelPass pass(std::move(ctx));

  g.init();

  // Always return false
  EXPECT_FALSE(pass.run(g.g()));

  EXPECT_EQ(loco::DataType::S16, g.input()->dtype());
}
