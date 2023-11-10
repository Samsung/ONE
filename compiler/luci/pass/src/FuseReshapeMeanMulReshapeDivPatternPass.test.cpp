/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseReshapeMeanMulReshapeDivPatternPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

/**
 *  Simple graph for test
 *
 *  BEFORE
 *                [Input]
 *              (3, 197, 197)
 *                  |
 *              [Reshape]
 *             (1, 591, 197)
 *                  |
 *          [CircleMean, axis<-1>]
 *              (1, 591, 1)
 *                  |
 *          [Mul, Scalar_Const]
 *              (1, 591, 1)
 *                  |
 *              [Reshape]
 *             (1, 3, 197, 1)
 *                 |
 *           [Div, Scalar_Const]
 *             (1, 3, 197, 1)
 *                 |
 *               [Output]
 *            (1, 3, 197, 1)
 *
 *  AFTER
 *                [Input]
 *              (3, 197, 197)
 *                  |
 *          [CircleMean, axis<-1>]
 *              (3, 197, 1)
 *                  |
 *              [Reshape]
 *             (1, 3, 197, 1)
 *                 |
 *          [Div, Scalar_Const]
 *             (1, 3, 197, 1)
 *                 |
 *               [Output]
 *            (1, 3, 197, 1)
 *
 */
class PatternReshapeMeanMulReshapeDivGraphlet
{
public:
  PatternReshapeMeanMulReshapeDivGraphlet() = default;

  void init(loco::Graph *g)
  {
    _reshape_1 = g->nodes()->create<luci::CircleReshape>();
    _reshape_const_1 = g->nodes()->create<luci::CircleConst>();

    _mean = g->nodes()->create<luci::CircleMean>();
    _mean_const = g->nodes()->create<luci::CircleConst>();

    _mul = g->nodes()->create<luci::CircleMul>();
    _mul_const = g->nodes()->create<luci::CircleConst>();

    _reshape_2 = g->nodes()->create<luci::CircleReshape>();
    _reshape_const_2 = g->nodes()->create<luci::CircleConst>();

    _div = g->nodes()->create<luci::CircleDiv>();
    _div_const = g->nodes()->create<luci::CircleConst>();

    _reshape_1->name("_reshape_1");
    _reshape_const_1->name("_reshape_const_1");

    _mean->name("_mean");
    _mean_const->name("_mean_const");

    _mul->name("_mul");
    _mul_const->name("_mul_const");

    _reshape_2->name("_reshape_2");
    _reshape_const_2->name("_reshape_const_2");

    _div->name("_div");
    _div_const->name("_div_const");
  }

public:
  luci::CircleReshape *reshape_1() { return _reshape_1; }
  luci::CircleConst *reshape_const_1() { return _reshape_const_1; }
  luci::CircleReshape *reshape_2() { return _reshape_2; }
  luci::CircleConst *reshape_const_2() { return _reshape_const_2; }
  luci::CircleMean *mean() { return _mean; }
  luci::CircleConst *mean_const() { return _mean_const; }
  luci::CircleMul *mul() { return _mul; }
  luci::CircleConst *mul_const() { return _mul_const; }
  luci::CircleDiv *div() { return _div; }
  luci::CircleConst *div_const() { return _div_const; }

protected:
  luci::CircleReshape *_reshape_1 = nullptr;
  luci::CircleConst *_reshape_const_1 = nullptr;
  luci::CircleMean *_mean = nullptr;
  luci::CircleConst *_mean_const = nullptr;
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_mul_const = nullptr;
  luci::CircleReshape *_reshape_2 = nullptr;
  luci::CircleConst *_reshape_const_2 = nullptr;
  luci::CircleDiv *_div = nullptr;
  luci::CircleConst *_div_const = nullptr;
};

class FuseReshapeMeanMulReshapeDivPatternTestGraph : public TestIOGraph,
                                                     public PatternReshapeMeanMulReshapeDivGraphlet
{
public:
  FuseReshapeMeanMulReshapeDivPatternTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({3, 197, 197}, {1, 3, 197, 1});
    PatternReshapeMeanMulReshapeDivGraphlet::init(g());

    _reshape_const_1->rank(1);
    _reshape_const_1->dtype(loco::DataType::S32);
    _reshape_const_1->size<loco::DataType::S32>(3);
    _reshape_const_1->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
    _reshape_const_1->at<loco::DataType::S32>(1) = static_cast<int32_t>(-1);
    _reshape_const_1->at<loco::DataType::S32>(2) = static_cast<int32_t>(197);
    _reshape_const_1->shape_status(luci::ShapeStatus::VALID);

    _reshape_const_2->rank(1);
    _reshape_const_2->dtype(loco::DataType::S32);
    _reshape_const_2->size<loco::DataType::S32>(4);
    _reshape_const_2->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
    _reshape_const_2->at<loco::DataType::S32>(1) = static_cast<int32_t>(-1);
    _reshape_const_2->at<loco::DataType::S32>(2) = static_cast<int32_t>(197);
    _reshape_const_2->at<loco::DataType::S32>(2) = static_cast<int32_t>(1);
    _reshape_const_2->shape_status(luci::ShapeStatus::VALID);

    _mean_const->rank(1);
    _mean_const->dtype(loco::DataType::S32);
    _mean_const->size<loco::DataType::S32>(1);
    _mean_const->at<loco::DataType::S32>(0) = static_cast<int32_t>(-1);
    _mean_const->shape_status(luci::ShapeStatus::VALID);

    _mul_const->rank(1);
    _mul_const->dtype(loco::DataType::FLOAT32);
    _mul_const->size<loco::DataType::FLOAT32>(1);
    _mul_const->at<loco::DataType::FLOAT32>(0) = 1.1f;
    _mul_const->shape_status(luci::ShapeStatus::VALID);

    _div_const->rank(1);
    _div_const->dtype(loco::DataType::FLOAT32);
    _div_const->size<loco::DataType::FLOAT32>(1);
    _div_const->at<loco::DataType::FLOAT32>(0) = 2.2f;
    _div_const->shape_status(luci::ShapeStatus::VALID);

    _reshape_1->rank(3);
    _reshape_1->dim(0).set(3);
    _reshape_1->dim(1).set(197);
    _reshape_1->dim(2).set(197);
    _reshape_1->dtype(loco::DataType::FLOAT32);
    _reshape_1->shape_status(luci::ShapeStatus::VALID);
    _reshape_1->tensor(input());
    _reshape_1->shape(_reshape_const_1);

    _mean->rank(3);
    _mean->dim(0).set(1);
    _mean->dim(1).set(591);
    _mean->dim(2).set(1);
    _mean->dtype(loco::DataType::FLOAT32);
    _mean->shape_status(luci::ShapeStatus::VALID);
    _mean->input(_reshape_1);
    _mean->reduction_indices(_mean_const);

    _mul->rank(3);
    _mul->dim(0).set(1);
    _mul->dim(1).set(591);
    _mul->dim(2).set(1);
    _mul->dtype(loco::DataType::FLOAT32);
    _mul->shape_status(luci::ShapeStatus::VALID);
    _mul->x(_mean);
    _mul->y(_mul_const);

    _reshape_2->rank(4);
    _reshape_2->dim(0).set(1);
    _reshape_2->dim(1).set(3);
    _reshape_2->dim(2).set(197);
    _reshape_2->dim(3).set(1);
    _reshape_2->dtype(loco::DataType::FLOAT32);
    _reshape_2->shape_status(luci::ShapeStatus::VALID);
    _reshape_2->tensor(_mul);
    _reshape_2->shape(_reshape_const_2);

    _div->rank(4);
    _div->dim(0).set(1);
    _div->dim(1).set(3);
    _div->dim(2).set(197);
    _div->dim(3).set(1);
    _div->dtype(loco::DataType::FLOAT32);
    _div->shape_status(luci::ShapeStatus::VALID);
    _div->x(_div_const);
    _div->y(_reshape_2);

    output()->from(_div);
  }
};

} // namespace

TEST(FuseReshapeMeanMulReshapeDivPatternPassTest, fuse_reshape_mean_mul_reshape_div)
{
  FuseReshapeMeanMulReshapeDivPatternTestGraph g;
  luci::FuseReshapeMeanMulReshapeDivPatternPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseReshapeMeanMulReshapeDivPatternPassTest, fuse_reshape_mean_mul_reshape_div_NEG)
{
  FuseReshapeMeanMulReshapeDivPatternTestGraph g;
  luci::FuseReshapeMeanMulReshapeDivPatternPass pass;

  g.init();

  // Add CircleRelu operation between CircleMean and Mul operations
  auto relu = g.g()->nodes()->create<luci::CircleRelu>();
  relu->name("relu");
  relu->features(g.mean());
  g.mul()->x(relu);

  EXPECT_FALSE(pass.run(g.g()));
}
