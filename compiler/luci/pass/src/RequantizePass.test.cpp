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

#include "luci/Pass/RequantizePass.h"

#include "helpers/CreateCircleConst.h"

#include <luci/test/TestIOGraph.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleQuantParam.h>

#include <vector>

#include <gtest/gtest.h>

using namespace luci;
using namespace luci::test;

namespace
{

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 * [IFM (S8)] [W (S8)] [B (S32)]
 *       |       |        |
 *       +-------+--------+
 *               |
 *               V
 *              [FC]
 *               |
 *               V
 *           [OFM(S8)]
 *
 *  AFTER
 *
 * [IFM (U8)] [W (U8)] [B (S32)]
 *       |       |        |
 *       +-------+--------+
 *               |
 *               V
 *              [FC]
 *               |
 *               V
 *           [OFM(U8)]
 */
struct S8FCGraphlet
{
public:
  S8FCGraphlet() = default;
  virtual ~S8FCGraphlet() = default;

  void init(loco::Graph *g, const ShapeU32 out_shape, const ShapeU32 w_shape,
            const ShapeU32 bias_shape)
  {
    _fc = g->nodes()->create<CircleFullyConnected>();
    _fc->input(_x);
    _x->dtype(loco::DataType::S8);
    {
      auto quantparam = std::make_unique<CircleQuantParam>();
      quantparam->scale.push_back(1.0);
      quantparam->zerop.push_back(0);
      quantparam->quantized_dimension = 0;
      _x->quantparam(std::move(quantparam));
    }

    _weights = create_const_node<int8_t>(g, loco::DataType::S8, w_shape, 1.0);
    {
      auto w_qparam = std::make_unique<CircleQuantParam>();
      std::vector<float> w_scale(_weights->dim(0).value(), 1.0);
      std::vector<int64_t> w_zp(_weights->dim(0).value(), 0);
      w_qparam->scale = w_scale;
      w_qparam->zerop = w_zp;
      w_qparam->quantized_dimension = 0;
      _weights->quantparam(std::move(w_qparam));
    }
    _fc->weights(_weights);

    _bias = create_const_node<int32_t>(g, loco::DataType::S32, bias_shape, 1.0);
    {
      auto b_qparam = std::make_unique<CircleQuantParam>();
      const auto bias_size = _bias->size<loco::DataType::S32>();
      std::vector<float> b_scale(bias_size, 1.0);
      std::vector<int64_t> b_zp(bias_size, 0);
      b_qparam->scale = b_scale;
      b_qparam->zerop = b_zp;
      b_qparam->quantized_dimension = 0;
      _bias->quantparam(std::move(b_qparam));
    }

    _fc->fusedActivationFunction(luci::FusedActFunc::NONE);
    _fc->dtype(loco::DataType::S8);
    _fc->shape(out_shape);
    _fc->bias(_bias);
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
  CircleFullyConnected *_fc = nullptr;
  CircleInput *_x = nullptr;
  CircleConst *_weights = nullptr;
  CircleConst *_bias = nullptr;
};

struct S8FCGraph final : public TestIGraphlet, public TestOGraphlet, public S8FCGraphlet
{
  void init(const ShapeU32 in_shape, const ShapeU32 w_shape, const ShapeU32 out_shape,
            const ShapeU32 bias_shape)
  {
    TestIGraphlet::init(g(), in_shape);
    TestOGraphlet::init(g(), out_shape);
    _x = input();
    S8FCGraphlet::init(g(), out_shape, w_shape, bias_shape);
    output()->from(_fc);
  }
};

class RequantizeS8ToU8FCTest : public ::testing::Test
{
public:
  S8FCGraph g;
};

} // namespace

TEST(RequantizePassTest, name)
{
  luci::RequantizePass pass(loco::DataType::FLOAT32, loco::DataType::U8);
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(RequantizeS8ToU8FCTest, FC)
{
  g.init({1, 18, 80} /* ifm shape */, {256, 80} /* weights shape*/, {18, 256} /* ofm shape */,
         {1, 256} /* bias shape*/);

  luci::RequantizePass rq(loco::DataType::S8, loco::DataType::U8);
  rq.run(g.g());

  EXPECT_EQ(loco::DataType::U8, g._x->dtype());
  EXPECT_EQ(loco::DataType::U8, g._fc->dtype());
  EXPECT_EQ(loco::DataType::U8, g._weights->dtype());
  EXPECT_EQ(loco::DataType::S32, g._bias->dtype());
}

TEST_F(RequantizeS8ToU8FCTest, FC_wrong_dtype_NEG)
{
  g.init({1, 18, 80} /* ifm shape */, {256, 80} /* weights shape*/, {18, 256} /* ofm shape */,
         {1, 256} /* bias shape*/);

  // Wrong dtype
  luci::RequantizePass rq(loco::DataType::U8, loco::DataType::S8);
  rq.run(g.g());

  EXPECT_EQ(loco::DataType::S8, g._x->dtype());
  EXPECT_EQ(loco::DataType::S8, g._fc->dtype());
  EXPECT_EQ(loco::DataType::S8, g._weights->dtype());
  EXPECT_EQ(loco::DataType::S32, g._bias->dtype());
}
