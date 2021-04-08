/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci/Pass/RemoveUnnecessarySlicePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>
#include "test/TestFirstNode.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class SliceGraphlet
{
public:
  SliceGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape, bool remove)
  {
    // Begin Create.
    _begin = g->nodes()->create<luci::CircleConst>();
    _begin->rank(1);
    _begin->dim(0).set(input_shape.size());
    _begin->shape_status(luci::ShapeStatus::VALID);
    _begin->dtype(loco::DataType::S32);
    _begin->size<loco::DataType::S32>(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
      _begin->at<loco::DataType::S32>(i) = remove ? 0 : 1;
    _begin->name("begin");

    // Size Create.
    _size = g->nodes()->create<luci::CircleConst>();
    _size->rank(1);
    _size->dim(0).set(input_shape.size());
    _size->shape_status(luci::ShapeStatus::VALID);
    _size->dtype(loco::DataType::S32);
    _size->size<loco::DataType::S32>(input_shape.size());
    for (int i = 0; i < input_shape.size(); ++i)
      _size->at<loco::DataType::S32>(i) = -1;
    _size->name("size");

    // Slice Node create.
    _slice = g->nodes()->create<luci::CircleSlice>();
    _slice->dtype(loco::DataType::S32);
    _slice->name("slice");
  }

protected:
  luci::CircleSlice *_slice = nullptr;
  luci::CircleConst *_begin = nullptr;
  luci::CircleConst *_size = nullptr;
};

class SliceGraph : public TestIOGraph, public SliceGraphlet
{
public:
  SliceGraph() = default;

public:
  void init(const ShapeU32 shape, bool remove)
  {
    TestIOGraph::init(shape, shape);
    SliceGraphlet::init(g(), shape, remove);

    _slice->input(input());
    _slice->begin(_begin);
    _slice->size(_size);

    output()->from(_slice);
  }
};

} // namespace

TEST(RemoveUnnecessarySlicePass, name)
{
  luci::RemoveUnnecessarySlicePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessarySlicePass, removed)
{
  SliceGraph g;

  g.init({2, 4, 2, 3}, true);

  // confirm graph has Slice
  auto slice_node = luci::test::first_node<luci::CircleSlice>(g.g());
  ASSERT_NE(nullptr, slice_node);
  luci::RemoveUnnecessarySlicePass pass;
  while (pass.run(g.g()))
    ;

  // check Slice is removed
  slice_node = luci::test::first_node<luci::CircleSlice>(g.g());
  ASSERT_EQ(nullptr, slice_node);
}

TEST(RemoveUnnecessarySlicePass, not_removed_NEG)
{
  SliceGraph g;

  g.init({2, 4, 2, 3}, false);

  // confirm graph has Slice
  auto slice_node = luci::test::first_node<luci::CircleSlice>(g.g());
  ASSERT_NE(nullptr, slice_node);
  luci::RemoveUnnecessarySlicePass pass;
  while (pass.run(g.g()))
    ;

  // check Slice is NOT removed
  slice_node = luci::test::first_node<luci::CircleSlice>(g.g());
  ASSERT_NE(nullptr, slice_node);
}
