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

#include "luci/Pass/FuseSliceWithTConvPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  TConv->Slice graph for test
 *
 *        [CircleInput]
 *              |
 *              |
 *     [CircleTransposeConv]
 *              |
 *              |
 *        [CircleSlice]
 *              |
 *              |
 *        [CircleOutput]
 */
struct TConvSliceGraph : public luci::test::TestIOGraph
{
  luci::CircleTransposeConv *_tconv = nullptr;
  luci::CircleSlice *_slice = nullptr;
  luci::CircleConst *_filter = nullptr;
  luci::CircleConst *_bias = nullptr;
  luci::CircleConst *_tconv_shape = nullptr;
  luci::CircleConst *_slice_offset = nullptr;
  luci::CircleConst *_slice_size = nullptr;

  TConvSliceGraph(uint32_t h, uint32_t w, uint32_t pads[4])
  {
    // pads={pad_top, pad_bottom, pad_left, pad_right}
    uint32_t channels = 32;
    uint32_t k_h = 3, k_w = 3;
    auto const tconv_h = (h - 1) * 2 + k_h;
    auto const tconv_w = (w - 1) * 2 + k_w;
    auto const out_h = tconv_h - pads[0] - pads[1];
    auto const out_w = tconv_w - pads[2] - pads[3];

    // graph input and output
    TestIOGraph::init({1, h, w, channels}, {1, out_h, out_w, channels});

    _filter = g()->nodes()->create<luci::CircleConst>();
    _filter->dtype(loco::DataType::FLOAT32);
    _filter->rank(4);
    _filter->shape({channels, k_h, k_w, channels});
    _filter->shape_status(luci::ShapeStatus::VALID);
    _filter->size<loco::DataType::FLOAT32>(channels * k_h * k_w * channels);
    _filter->name("filter");

    _bias = g()->nodes()->create<luci::CircleConst>();
    _bias->dtype(loco::DataType::FLOAT32);
    _bias->rank(1);
    _bias->shape({channels});
    _bias->shape_status(luci::ShapeStatus::VALID);
    _bias->size<loco::DataType::FLOAT32>(channels);
    _bias->name("bias");

    _tconv_shape = g()->nodes()->create<luci::CircleConst>();
    _tconv_shape->dtype(loco::DataType::S32);
    _tconv_shape->rank(1);
    _tconv_shape->shape({4});
    _tconv_shape->shape_status(luci::ShapeStatus::VALID);
    _tconv_shape->size<loco::DataType::S32>(4);
    _tconv_shape->at<loco::DataType::S32>(0) = 1;
    _tconv_shape->at<loco::DataType::S32>(3) = channels;
    _tconv_shape->at<loco::DataType::S32>(1) = tconv_h;
    _tconv_shape->at<loco::DataType::S32>(2) = tconv_w;
    _tconv_shape->name("tconv_shape");

    _tconv = g()->nodes()->create<luci::CircleTransposeConv>();
    _tconv->filter(_filter);
    _tconv->bias(_bias);
    _tconv->inputSizes(_tconv_shape);
    _tconv->outBackprop(input());
    _tconv->fusedActivationFunction(luci::FusedActFunc::NONE);
    _tconv->dtype(loco::DataType::FLOAT32);
    _tconv->padding(luci::Padding::VALID);
    _tconv->stride()->h(2);
    _tconv->stride()->w(2);
    _tconv->name("tconv");

    // offset to be used in slice
    _slice_offset = g()->nodes()->create<luci::CircleConst>();
    _slice_offset->dtype(loco::DataType::S32);
    _slice_offset->rank(1);
    _slice_offset->shape({4});
    _slice_offset->shape_status(luci::ShapeStatus::VALID);
    _slice_offset->size<loco::DataType::S32>(4);
    _slice_offset->at<loco::DataType::S32>(0) = 0;
    _slice_offset->at<loco::DataType::S32>(3) = 0;
    _slice_offset->at<loco::DataType::S32>(1) = pads[0];
    _slice_offset->at<loco::DataType::S32>(2) = pads[2];
    _slice_offset->name("slice_offset");

    _slice_size = g()->nodes()->create<luci::CircleConst>();
    _slice_size->dtype(loco::DataType::S32);
    _slice_size->rank(1);
    _slice_size->shape({4});
    _slice_size->shape_status(luci::ShapeStatus::VALID);
    _slice_size->size<loco::DataType::S32>(4);
    _slice_size->at<loco::DataType::S32>(0) = 1;
    _slice_size->at<loco::DataType::S32>(3) = channels;
    _slice_size->at<loco::DataType::S32>(1) = out_h;
    _slice_size->at<loco::DataType::S32>(2) = out_w;
    _slice_size->name("slice_size");

    _slice = g()->nodes()->create<luci::CircleSlice>();
    _slice->begin(_slice_offset);
    _slice->size(_slice_size);
    _slice->input(_tconv);
    _slice->name("slice");

    output()->from(_slice);
  }
};

} // namespace

TEST(FuseSliceWithTConvPassTest, simple_test)
{
  /**
   *  tests:
   *    1) fusion pass has nonnull name
   *    2) fusion runs successfully for float32 TConvSlice graph
   *    3) resulting graph has the following structure:
   *
   *      [CircleTransposeConv] (with output_shape = shape_of_the_slice)
   *              |
   *              |
   *           [Output]
   */
  luci::FuseSliceWithTConvPass pass;
  uint32_t pads[4] = {0, 2, 0, 2};
  uint32_t h = 8, w = 8;
  TConvSliceGraph graph(h, w, pads);
  auto const out_h = graph._slice_size->at<loco::DataType::S32>(1);
  auto const out_w = graph._slice_size->at<loco::DataType::S32>(2);

  auto const name = pass.name();
  ASSERT_NE(nullptr, name);

  auto ret = pass.run(graph.g());
  EXPECT_TRUE(ret);

  auto const fused_tconv = dynamic_cast<luci::CircleTransposeConv *>(graph.output()->from());
  EXPECT_NE(nullptr, fused_tconv);

  EXPECT_EQ(luci::Padding::VALID, fused_tconv->padding());

  auto const out_size = dynamic_cast<luci::CircleConst *>(fused_tconv->inputSizes());
  EXPECT_NE(nullptr, out_size);
  EXPECT_EQ(out_h, out_size->at<loco::DataType::S32>(1)); // h
  EXPECT_EQ(out_w, out_size->at<loco::DataType::S32>(2)); // 2
}

TEST(FuseSliceWithTConvPassTest, wrong_condition_NEG)
{
  luci::FuseSliceWithTConvPass pass;
  uint32_t pads[4] = {3, 3, 3, 3}; // no fusion is possible with these pads
  TConvSliceGraph graph(8, 8, pads);

  auto ret = pass.run(graph.g());
  EXPECT_FALSE(ret);
}
