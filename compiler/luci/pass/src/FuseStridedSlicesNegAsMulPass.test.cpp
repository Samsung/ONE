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

#include "luci/Pass/FuseStridedSlicesNegAsMulPass.h"

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
 *                            |
 *                       [CircleNode]
 *                      |            |
 *        [CircleStridedSlice]  [CircleStridedSlice]
 *                     |             |
 *                     |        [CircleNeg]
 *                     |            |
 *                  [CircleConcatenation]
 *                            |
 *
 *  AFTER
 *                            |
 *                       [CircleNode]
 *                            |
 *                       [CircleMul] ------- [CircleConst]
 *                            |
 *
 *
 */
class FuseStridedSlicesNegAsMulTestGraph : public TestIOGraph
{
public:
  FuseStridedSlicesNegAsMulTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 10}, {1, 10});

    _concat = g()->nodes()->create<luci::CircleConcatenation>(2);
    _neg = g()->nodes()->create<luci::CircleNeg>();
    _ss_with_neg = g()->nodes()->create<luci::CircleStridedSlice>();
    _ss_without_neg = g()->nodes()->create<luci::CircleStridedSlice>();

    _begin_ss_with_neg = g()->nodes()->create<luci::CircleConst>();
    _end_ss_with_neg = g()->nodes()->create<luci::CircleConst>();
    _strides_ss_with_neg = g()->nodes()->create<luci::CircleConst>();

    _begin_ss_without_neg = g()->nodes()->create<luci::CircleConst>();
    _end_ss_without_neg = g()->nodes()->create<luci::CircleConst>();
    _strides_ss_without_neg = g()->nodes()->create<luci::CircleConst>();

    _concat->name("concat");
    _neg->name("neg");
    _ss_with_neg->name("strided_slice_with_neg");
    _ss_without_neg->name("strided_slice_without_neg");

    // StridedSlice consts with neg
    _begin_ss_with_neg->rank(2);
    _begin_ss_with_neg->dtype(loco::DataType::S32);
    _begin_ss_with_neg->size<loco::DataType::S32>(2);
    _begin_ss_with_neg->at<loco::DataType::S32>(0) = static_cast<int32_t>(0);
    _begin_ss_with_neg->at<loco::DataType::S32>(1) = static_cast<int32_t>(0);
    _begin_ss_with_neg->dim(0) = 2;
    _begin_ss_with_neg->shape_status(luci::ShapeStatus::VALID);

    _end_ss_with_neg->rank(2);
    _end_ss_with_neg->dtype(loco::DataType::S32);
    _end_ss_with_neg->size<loco::DataType::S32>(2);
    _end_ss_with_neg->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
    _end_ss_with_neg->at<loco::DataType::S32>(1) = static_cast<int32_t>(5);
    _end_ss_with_neg->dim(0) = 2;
    _end_ss_with_neg->shape_status(luci::ShapeStatus::VALID);

    _strides_ss_with_neg->rank(2);
    _strides_ss_with_neg->dtype(loco::DataType::S32);
    _strides_ss_with_neg->size<loco::DataType::S32>(2);
    _strides_ss_with_neg->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
    _strides_ss_with_neg->at<loco::DataType::S32>(1) = static_cast<int32_t>(1);
    _strides_ss_with_neg->dim(0) = 2;
    _strides_ss_with_neg->shape_status(luci::ShapeStatus::VALID);

    // StridedSlice consts without neg
    _begin_ss_without_neg->rank(2);
    _begin_ss_without_neg->dtype(loco::DataType::S32);
    _begin_ss_without_neg->size<loco::DataType::S32>(2);
    _begin_ss_without_neg->at<loco::DataType::S32>(0) = static_cast<int32_t>(0);
    _begin_ss_without_neg->at<loco::DataType::S32>(1) = static_cast<int32_t>(5);
    _begin_ss_without_neg->dim(0) = 2;
    _begin_ss_without_neg->shape_status(luci::ShapeStatus::VALID);

    _end_ss_without_neg->rank(2);
    _end_ss_without_neg->dtype(loco::DataType::S32);
    _end_ss_without_neg->size<loco::DataType::S32>(2);
    _end_ss_without_neg->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
    _end_ss_without_neg->at<loco::DataType::S32>(1) = static_cast<int32_t>(10);
    _end_ss_without_neg->dim(0) = 2;
    _end_ss_without_neg->shape_status(luci::ShapeStatus::VALID);

    _strides_ss_without_neg->rank(2);
    _strides_ss_without_neg->dtype(loco::DataType::S32);
    _strides_ss_without_neg->size<loco::DataType::S32>(2);
    _strides_ss_without_neg->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
    _strides_ss_without_neg->at<loco::DataType::S32>(1) = static_cast<int32_t>(1);
    _strides_ss_without_neg->dim(0) = 2;
    _strides_ss_without_neg->shape_status(luci::ShapeStatus::VALID);

    _concat->values(0, _neg);
    _concat->values(1, _ss_without_neg);
    _concat->rank(2);
    _concat->dim(0) = 1;
    _concat->dim(1) = 10;

    _neg->x(_ss_with_neg);

    _ss_without_neg->input(input());
    _ss_without_neg->strides(_strides_ss_without_neg);
    _ss_without_neg->begin(_begin_ss_without_neg);
    _ss_without_neg->end(_end_ss_without_neg);

    _ss_with_neg->input(input());
    _ss_with_neg->begin(_begin_ss_with_neg);
    _ss_with_neg->end(_end_ss_with_neg);
    _ss_with_neg->strides(_strides_ss_with_neg);

    _concat->dtype(loco::DataType::FLOAT32);
    _ss_with_neg->dtype(loco::DataType::FLOAT32);
    _neg->dtype(loco::DataType::FLOAT32);
    _ss_without_neg->dtype(loco::DataType::FLOAT32);

    output()->from(_concat);
  }

  luci::CircleNeg *neg() { return _neg; }
  luci::CircleConcatenation *concat() { return _concat; }

private:
  luci::CircleConcatenation *_concat = nullptr;
  luci::CircleNeg *_neg = nullptr;
  luci::CircleStridedSlice *_ss_with_neg = nullptr;
  luci::CircleStridedSlice *_ss_without_neg = nullptr;

  luci::CircleConst *_begin_ss_with_neg = nullptr;
  luci::CircleConst *_end_ss_with_neg = nullptr;
  luci::CircleConst *_strides_ss_with_neg = nullptr;

  luci::CircleConst *_begin_ss_without_neg = nullptr;
  luci::CircleConst *_end_ss_without_neg = nullptr;
  luci::CircleConst *_strides_ss_without_neg = nullptr;
};

} // namespace

TEST(FuseStridedSlicesNegAsMulPassTest, name)
{
  luci::FuseStridedSlicesNegAsMulPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseStridedSlicesNegAsMulPassTest, fuse_strided_slices_neg_as_mul)
{
  FuseStridedSlicesNegAsMulTestGraph g;
  luci::FuseStridedSlicesNegAsMulPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));

  auto mul = dynamic_cast<luci::CircleMul *>(g.output()->from());
  EXPECT_NE(nullptr, mul);
}

TEST(FuseStridedSlicesNegAsMulPassTest, fuse_strided_slices_neg_as_mul_NEG)
{
  FuseStridedSlicesNegAsMulTestGraph g;
  luci::FuseStridedSlicesNegAsMulPass pass;

  g.init();

  // Add CircleRelu operation between CircleNeg and CircleConcatenation
  auto relu = g.g()->nodes()->create<luci::CircleRelu>();
  relu->name("relu");
  relu->features(g.neg());
  g.concat()->values(0, relu);

  // Due to the CircleRelu operation, pass will not be applied
  EXPECT_FALSE(pass.run(g.g()));
}
