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

#include "QuantizeBias.h"

#include "helpers/CreateCircleConst.h"

#include <luci/test/TestIOGraph.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleQuantParam.h>

#include <gtest/gtest.h>

using namespace luci;

namespace
{

using namespace luci::test;

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *   [IFM] [WEIGHTS] [BIAS(FP32)]
 *        \   |     /
 *           [FC]
 *            |
 *          [OFM]
 *
 *  AFTER
 *
 *   [IFM] [WEIGHTS] [BIAS(Quantized)]
 *        \   |     /
 *           [FC]
 *            |
 *          [OFM]
 */
struct Q8FCGraphlet
{
public:
  Q8FCGraphlet() = default;
  virtual ~Q8FCGraphlet() = default;

  void init(loco::Graph *g, const ShapeU32 out_shape, const ShapeU32 w_shape,
            const ShapeU32 bias_shape, const float bv)
  {
    _fc = g->nodes()->create<luci::CircleFullyConnected>();
    _fc->input(_x);
    _x->dtype(loco::DataType::U8);
    {
      auto quantparam = std::make_unique<CircleQuantParam>();
      quantparam->scale.push_back(1.0);
      quantparam->zerop.push_back(0);
      quantparam->quantized_dimension = 0;
      _x->quantparam(std::move(quantparam));
    }

    auto weights = create_const_node<uint8_t>(g, loco::DataType::U8, w_shape, 1.0);
    auto w_qparam = std::make_unique<CircleQuantParam>();
    std::vector<float> w_scale(weights->dim(0).value(), 1.0);
    std::vector<int64_t> w_zp(weights->dim(0).value(), 0);
    w_qparam->scale = w_scale;
    w_qparam->zerop = w_zp;
    w_qparam->quantized_dimension = 0;
    weights->quantparam(std::move(w_qparam));
    _fc->weights(weights);
    _fc->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc->dtype(loco::DataType::U8);
    _fc->shape(out_shape);
    auto l = _fc->dim(_fc->rank() - 1).value();
    _fc->bias(create_const_node(g, loco::DataType::FLOAT32, bias_shape, bv));
    _fc->name("fc");
    {
      auto quantparam = std::make_unique<CircleQuantParam>();
      quantparam->scale.push_back(1.0);
      quantparam->zerop.push_back(0);
      quantparam->quantized_dimension = 0;
      _fc->quantparam(std::move(quantparam));
    }
  }

public:
  luci::CircleFullyConnected *fc() { return _fc; }

protected:
  luci::CircleFullyConnected *_fc = nullptr;
  luci::CircleInput *_x = nullptr;
};

struct Q8FCGraph final : public TestIGraphlet, public TestOGraphlet, public Q8FCGraphlet
{
  void init(const ShapeU32 in_shape, const ShapeU32 w_shape, const ShapeU32 out_shape,
            const ShapeU32 bias_shape, const float bv)
  {
    TestIGraphlet::init(g(), in_shape);
    TestOGraphlet::init(g(), out_shape);
    _x = input();
    Q8FCGraphlet::init(g(), out_shape, w_shape, bias_shape, bv);
    output()->from(_fc);
  }
};

class CQ8QuantizeBiasFCTest : public ::testing::Test
{
public:
  Q8FCGraph g;
  luci::QuantizeBias qb{loco::DataType::FLOAT32, loco::DataType::U8,
                        luci::QuantizationGranularity::ChannelWise};
};

} // namespace

TEST_F(CQ8QuantizeBiasFCTest, fully_connected)
{
  g.init({1, 18, 80}, {256, 80}, {18, 256}, {1, 256}, 1);
  g.fc()->accept(&qb);

  auto bias = loco::must_cast<CircleConst *>(g.fc()->bias());
  auto qparam = bias->quantparam();

  EXPECT_NE(nullptr, qparam);
  EXPECT_EQ(256, qparam->scale.size());
  EXPECT_EQ(256, qparam->zerop.size());
  EXPECT_EQ(1, qparam->quantized_dimension);
}

TEST_F(CQ8QuantizeBiasFCTest, wrong_bias_shape_NEG)
{
  g.init({1, 18, 80}, {256, 80}, {18, 256}, {1, 2, 128}, 1);
  EXPECT_ANY_THROW(g.fc()->accept(&qb)); // Wrong bias shape
}
