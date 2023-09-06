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

#include "Quantizer.h"
#include "TestHelper.h"

#include <luci/IR/CircleNodes.h>

#include <cmath>

namespace
{

class AddGraph final : public SimpleGraph
{
protected:
  void initInput(loco::Node *input) override
  {
    auto ci_input = loco::must_cast<luci::CircleNode *>(input);
    initMinMax(ci_input);
  }

  void initMinMax(luci::CircleNode *node)
  {
    auto qparam = std::make_unique<luci::CircleQuantParam>();
    qparam->min.assign(1, _a_min);
    qparam->max.assign(1, _a_max);
    node->quantparam(std::move(qparam));
  }

  loco::Node *insertGraphBody(loco::Node *input) override
  {
    _add = _g->nodes()->create<luci::CircleAdd>();
    _beta = _g->nodes()->create<luci::CircleConst>();

    _add->dtype(loco::DataType::FLOAT32);
    _beta->dtype(loco::DataType::FLOAT32);

    _add->shape({1, _channel_size, _width, _height});
    _beta->shape({1, _channel_size, _width, _height});

    _beta->size<loco::DataType::FLOAT32>(_channel_size * _width * _height);
    _add->x(input);
    _add->y(_beta);
    _add->fusedActivationFunction(luci::FusedActFunc::NONE);

    _add->name("add");
    _beta->name("beta");
    initMinMax(_add);

    return _add;
  }

public:
  float _a_min = -1.f;
  float _a_max = 1.f;
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_beta = nullptr;
};

} // namespace

TEST(CircleMPQSolverQuantizerTest, verifyResultsTest)
{
  auto m = luci::make_module();
  AddGraph g;
  g.init();
  auto add = g._add;
  float range = g._a_max - g._a_min;
  g.transfer_to(m.get());

  std::string def_quant = "uint8";
  mpqsolver::core::Quantizer quantizer(def_quant, def_quant);
  mpqsolver::core::LayerParams params;
  auto res = quantizer.quantize(m.get(), def_quant, params);
  EXPECT_TRUE(res);
  auto quant_param = add->quantparam();
  EXPECT_TRUE(quant_param != nullptr);
  EXPECT_TRUE(quant_param->scale.size() == 1);
  EXPECT_FLOAT_EQ(quant_param->scale[0], range / 255.f);
  EXPECT_TRUE(quant_param->zerop.size() == 1);
  EXPECT_TRUE(quant_param->zerop[0] == 128);
}

TEST(CircleMPQSolverQuantizerTest, verifyResultsTest_NEG)
{
  std::string def_quant = "uint8";
  mpqsolver::core::Quantizer quantizer(def_quant, def_quant);
  mpqsolver::core::LayerParams params;
  auto res = quantizer.quantize(nullptr, def_quant, params);
  EXPECT_TRUE(!res);
}
