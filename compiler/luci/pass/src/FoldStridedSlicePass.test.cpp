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

#include "luci/Pass/FoldStridedSlicePass.h"
#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Graph has an StridedSlice Op with constant inputs
 *
 * BEFORE
 *
 *        [CircleConst]
 *              |
 *     [CircleStridedSlice]
 *              |
 *       [CircleOutput]
 *
 * AFTER
 *
 *       [CircleConst]
 *
 */
class FoldStridedSliceTest : public luci::ConstantFoldingTestGraph, public ::testing::Test
{
public:
  FoldStridedSliceTest() : luci::ConstantFoldingTestGraph({1, 4, 4, 1}, loco::DataType::S32)
  {
    _strided_slice = _g.nodes()->create<luci::CircleStridedSlice>();
    _strided_slice_input = _g.nodes()->create<luci::CircleConst>();
    _strided_slice_begin = _g.nodes()->create<luci::CircleConst>();
    _strided_slice_end = _g.nodes()->create<luci::CircleConst>();
    _strided_slice_strides = _g.nodes()->create<luci::CircleConst>();

    _strided_slice->dtype(loco::DataType::S32);
    _strided_slice->shape({1, 4, 4, 1});
    _strided_slice->shape_status(luci::ShapeStatus::VALID);
    _strided_slice->input(_strided_slice_input);
    _strided_slice->begin(_strided_slice_begin);
    _strided_slice->end(_strided_slice_end);
    _strided_slice->strides(_strided_slice_strides);
    _strided_slice->begin_mask(0);
    _strided_slice->end_mask(0);
    _strided_slice->ellipsis_mask(0);
    _strided_slice->new_axis_mask(0);
    _strided_slice->shrink_axis_mask(0);

    _strided_slice_input->name("strided_slice_input");
    _strided_slice_input->dtype(loco::DataType::S32);
    _strided_slice_input->shape({1, 4, 4, 1});
    _strided_slice_input->size<loco::DataType::S32>(16);

    _strided_slice_begin->dtype(loco::DataType::S32);
    _strided_slice_begin->shape({4});
    _strided_slice_begin->size<loco::DataType::S32>(4);

    _strided_slice_end->dtype(loco::DataType::S32);
    _strided_slice_end->shape({4});
    _strided_slice_end->size<loco::DataType::S32>(4);

    _strided_slice_strides->dtype(loco::DataType::S32);
    _strided_slice_strides->shape({4});
    _strided_slice_strides->size<loco::DataType::S32>(4);

    _output->from(_strided_slice);
  }

protected:
  void init() final {}

protected:
  loco::Node *createFoldedPattern() final { return nullptr; }

protected:
  luci::CircleConst *getFoldedPattern() final
  {
    return loco::must_cast<luci::CircleConst *>(_output->from());
  }

protected:
  luci::CircleStridedSlice *_strided_slice = nullptr;
  luci::CircleConst *_strided_slice_input = nullptr;
  luci::CircleConst *_strided_slice_begin = nullptr;
  luci::CircleConst *_strided_slice_end = nullptr;
  luci::CircleConst *_strided_slice_strides = nullptr;
};

} // namespace

TEST(FoldStridedSlicePass, name)
{
  luci::FoldStridedSlicePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(FoldStridedSliceTest, fold_strided_slice)
{
  for (uint32_t i = 0; i < 16; ++i)
    _strided_slice_input->at<loco::DataType::S32>(i) = i;
  _strided_slice_begin->at<loco::DataType::S32>(0) = 0;
  _strided_slice_begin->at<loco::DataType::S32>(1) = 0;
  _strided_slice_begin->at<loco::DataType::S32>(2) = 0;
  _strided_slice_begin->at<loco::DataType::S32>(3) = 0;
  _strided_slice_end->at<loco::DataType::S32>(0) = 1;
  _strided_slice_end->at<loco::DataType::S32>(1) = 4;
  _strided_slice_end->at<loco::DataType::S32>(2) = 4;
  _strided_slice_end->at<loco::DataType::S32>(3) = 1;
  _strided_slice_strides->at<loco::DataType::S32>(0) = 1;
  _strided_slice_strides->at<loco::DataType::S32>(1) = 2;
  _strided_slice_strides->at<loco::DataType::S32>(2) = 2;
  _strided_slice_strides->at<loco::DataType::S32>(3) = 1;

  luci::FoldStridedSlicePass pass;
  ASSERT_TRUE(pass.run(&_g));

  auto folded_const = getFoldedPattern();
  EXPECT_EQ(folded_const->dtype(), loco::DataType::S32);
  EXPECT_EQ(folded_const->at<loco::DataType::S32>(0), 0);
  EXPECT_EQ(folded_const->at<loco::DataType::S32>(1), 2);
  EXPECT_EQ(folded_const->at<loco::DataType::S32>(2), 8);
  EXPECT_EQ(folded_const->at<loco::DataType::S32>(3), 10);
}

TEST_F(FoldStridedSliceTest, fold_non_constant_NEG)
{
  _strided_slice->input(_input);

  luci::FoldStridedSlicePass pass;
  ASSERT_FALSE(pass.run(&_g));
}
