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

#include "BatchNormPatternFinder.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace luci
{
namespace test
{

/**
 * @brief Graphlet with Add and Const as beta from BatchNorm
 */
class AddBetaGraphlet
{
public:
  AddBetaGraphlet() = default;

  void init(loco::Graph *g, const ShapeU32 shape, luci::FusedActFunc actf)
  {
    _add = g->nodes()->create<luci::CircleAdd>();
    _add_beta = g->nodes()->create<luci::CircleConst>();

    _add->dtype(loco::DataType::FLOAT32);
    _add_beta->dtype(loco::DataType::FLOAT32);

    _add->fusedActivationFunction(actf);

    assert(shape.size() > 0);
    auto last_it = std::prev(shape.end(), 1);
    auto channel_size = *last_it;

    _add->shape(shape);
    _add_beta->shape({channel_size});
    _add_beta->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
      _add_beta->at<loco::DataType::FLOAT32>(i) = i;

    _add->name("add");
    _add_beta->name("add_beta");
  }

public:
  luci::CircleAdd *add() { return _add; }

protected:
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_add_beta = nullptr;
};

/**
 * @brief Graphlet with Mul and Const as gamma from BatchNorm
 */
class MulGammaGraphlet
{
public:
  MulGammaGraphlet() = default;

  void init(loco::Graph *g, const ShapeU32 shape, luci::FusedActFunc actf)
  {
    _mul = g->nodes()->create<luci::CircleMul>();
    _mul_gamma = g->nodes()->create<luci::CircleConst>();

    _mul->dtype(loco::DataType::FLOAT32);
    _mul_gamma->dtype(loco::DataType::FLOAT32);

    _mul->fusedActivationFunction(actf);

    assert(shape.size() > 0);
    auto last_it = std::prev(shape.end(), 1);
    auto channel_size = *last_it;

    _mul->shape(shape);
    _mul_gamma->shape({channel_size});
    _mul_gamma->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
      _mul_gamma->at<loco::DataType::FLOAT32>(i) = i;

    _mul->name("mul");
    _mul_gamma->name("mul_gamma");
  }

public:
  luci::CircleMul *mul(void) { return _mul; }

protected:
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_mul_gamma = nullptr;
};

/**
 * @brief Graph of Mul-Add pattern from BatchNorm
 */
class MulAddGraph : public TestIOGraph, public AddBetaGraphlet, public MulGammaGraphlet
{
public:
  MulAddGraph() = default;

  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    MulGammaGraphlet::init(g(), shape_in, luci::FusedActFunc::NONE);
    AddBetaGraphlet::init(g(), shape_out, luci::FusedActFunc::RELU);

    // connect network
    _mul->x(input());
    _mul->y(_mul_gamma);
    _add->x(_mul);
    _add->y(_add_beta);
    output()->from(_add);
  }
};

/**
 * @brief Graph of Add with Const
 */
class AddGraph : public TestIOGraph, public AddBetaGraphlet
{
public:
  AddGraph() = default;

  void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);
    AddBetaGraphlet::init(g(), shape_in, luci::FusedActFunc::RELU);

    // connect network
    _add->x(input());
    _add->y(_add_beta);
    output()->from(_add);
  }
};

} // namespace test
} // namespace luci

class BatchNormPatternFinderMulAddTest : public ::testing::Test
{
public:
  BatchNormPatternFinderMulAddTest() = default;

protected:
  luci::test::MulAddGraph _mag;
};

class BatchNormPatternFinderAddTest : public ::testing::Test
{
public:
  BatchNormPatternFinderAddTest() = default;

protected:
  luci::test::AddGraph _ag;
};

TEST_F(BatchNormPatternFinderMulAddTest, is_batchnorm_add)
{
  _mag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  luci::CircleMul *mul = nullptr;
  luci::CircleConst *beta = nullptr;

  auto res = luci::is_batchnorm_add(_mag.add(), mul, beta);
  ASSERT_TRUE(res);
  ASSERT_NE(nullptr, mul);
  ASSERT_NE(nullptr, beta);
}

TEST_F(BatchNormPatternFinderMulAddTest, is_batchnorm_add2)
{
  _mag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  auto res = luci::is_batchnorm_add(_mag.add());
  ASSERT_TRUE(res);
}

TEST_F(BatchNormPatternFinderAddTest, is_batchnorm_add_NEG)
{
  _ag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  luci::CircleMul *mul = nullptr;
  luci::CircleConst *beta = nullptr;

  auto res = luci::is_batchnorm_add(_ag.add(), mul, beta);
  ASSERT_FALSE(res);
}

TEST_F(BatchNormPatternFinderMulAddTest, is_batchnorm_mul)
{
  _mag.init({1, 16, 16, 4}, {1, 16, 16, 4});

  luci::CircleNode *pred = nullptr;
  luci::CircleConst *gamma = nullptr;

  auto res = luci::is_batchnorm_mul(_mag.mul(), pred, gamma);
  ASSERT_TRUE(res);
  ASSERT_NE(nullptr, pred);
  ASSERT_NE(nullptr, gamma);
}
