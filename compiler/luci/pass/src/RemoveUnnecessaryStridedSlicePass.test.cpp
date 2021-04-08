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
#include "luci/Pass/RemoveUnnecessaryStridedSlicePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>
#include "test/TestFirstNode.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class StridedSliceGraphlet
{
public:
  StridedSliceGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape, bool remove)
  {
    // Begin create
    _begin = g->nodes()->create<luci::CircleConst>();
    _begin->rank(1);
    _begin->dim(0).set(input_shape.size());
    _begin->shape_status(luci::ShapeStatus::VALID);
    _begin->dtype(loco::DataType::S32);
    _begin->size<loco::DataType::S32>(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
    {
      _begin->at<loco::DataType::S32>(i) = remove ? 0 : 1;
    }

    // Strides create
    _strides = g->nodes()->create<luci::CircleConst>();
    _strides->rank(1);
    _strides->dim(0).set(input_shape.size());
    _strides->shape_status(luci::ShapeStatus::VALID);
    _strides->dtype(loco::DataType::S32);
    _strides->size<loco::DataType::S32>(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
    {
      _strides->at<loco::DataType::S32>(i) = remove ? 1 : -1;
    }

    std::vector<uint32_t> shape_vector{input_shape};

    _end = g->nodes()->create<luci::CircleConst>();
    _end->rank(1);
    _end->dim(0).set(input_shape.size());
    _end->shape_status(luci::ShapeStatus::VALID);
    _end->dtype(loco::DataType::S32);
    _end->size<loco::DataType::S32>(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
    {
      if (remove)
        _end->at<loco::DataType::S32>(i) = static_cast<int32_t>(shape_vector.at(i));
      else
        _end->at<loco::DataType::S32>(i) = -1;
    }

    // StridedSlice Node create
    _strided_slice = g->nodes()->create<luci::CircleStridedSlice>();
    _strided_slice->dtype(loco::DataType::S32);
  }

protected:
  luci::CircleStridedSlice *_strided_slice = nullptr;
  luci::CircleConst *_begin = nullptr;
  luci::CircleConst *_strides = nullptr;
  luci::CircleConst *_end = nullptr;
};

class StridedSliceGraph : public TestIOGraph, public StridedSliceGraphlet
{
public:
  StridedSliceGraph() = default;

public:
  void init(const ShapeU32 shape, bool remove)
  {
    TestIOGraph::init(shape, shape);
    StridedSliceGraphlet::init(g(), shape, remove);

    _strided_slice->input(input());
    _strided_slice->begin(_begin);
    _strided_slice->strides(_strides);
    _strided_slice->end(_end);

    output()->from(_strided_slice);
  }
};

} // namespace

TEST(RemoveUnnecessaryStridedSlicePass, basic_case)
{
  StridedSliceGraph g;

  g.init({2, 4, 2, 3}, true);

  auto strided_slice_node = luci::test::first_node<luci::CircleStridedSlice>(g.g());
  ASSERT_NE(nullptr, strided_slice_node);
  luci::RemoveUnnecessaryStridedSlicePass pass;
  while (pass.run(g.g()))
    ;

  strided_slice_node = luci::test::first_node<luci::CircleStridedSlice>(g.g());
  ASSERT_EQ(nullptr, strided_slice_node);
}

TEST(RemoveUnnecessaryStridedSlicePass, basic_fail_case_NEG)
{
  StridedSliceGraph g;

  g.init({2, 4, 2, 3}, false);

  auto strided_slice_node = luci::test::first_node<luci::CircleStridedSlice>(g.g());
  ASSERT_NE(nullptr, strided_slice_node);
  luci::RemoveUnnecessaryStridedSlicePass pass;
  while (pass.run(g.g()))
    ;

  strided_slice_node = luci::test::first_node<luci::CircleStridedSlice>(g.g());
  ASSERT_NE(nullptr, strided_slice_node);
}
