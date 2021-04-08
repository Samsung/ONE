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

#include "luci/Pass/RemoveUnnecessaryReshapePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>
#include "test/TestFirstNode.h"

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class ReshapeGraphlet
{
public:
  ReshapeGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape, bool remove)
  {
    std::vector<uint32_t> shape_vector{input_shape};

    auto dim0_val = remove ? shape_vector.size() : 1;
    _reshape_shape = g->nodes()->create<luci::CircleConst>();
    _reshape_shape->rank(1);
    _reshape_shape->dim(0).set(dim0_val);
    _reshape_shape->shape_status(luci::ShapeStatus::VALID);
    _reshape_shape->dtype(loco::DataType::S32);

    _reshape_shape->size<loco::DataType::S32>(dim0_val);
    for (uint32_t i = 0; i < dim0_val; i++)
    {
      if (remove)
        _reshape_shape->at<loco::DataType::S32>(i) = static_cast<int32_t>(shape_vector.at(i));
      else
        _reshape_shape->at<loco::DataType::S32>(i) = -1;
    }
    _reshape_shape->name("reshape_shape");

    // Reshape create
    auto newshape_rank = remove ? shape_vector.size() : 1;
    _reshape = g->nodes()->create<luci::CircleReshape>();
    _reshape->newShape()->rank(newshape_rank);
    for (uint32_t i = 0; i < newshape_rank; i++)
    {
      if (remove)
        _reshape->newShape()->dim(i) = static_cast<int32_t>(shape_vector.at(i));
      else
        _reshape->newShape()->dim(i) = -1;
    }
    _reshape->name("reshape");
  }

protected:
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleConst *_reshape_shape = nullptr;
};

class ReshapeGraph : public TestIOGraph, public ReshapeGraphlet
{
public:
  ReshapeGraph() = default;

public:
  void init(const ShapeU32 shape, bool remove)
  {
    TestIOGraph::init(shape, shape);
    ReshapeGraphlet::init(g(), shape, remove);

    // connect graph
    _reshape->tensor(input());
    _reshape->shape(_reshape_shape);

    output()->from(_reshape);
  }
};

// TODO use ::testing::Test

} // namespace

TEST(RemoveUnnecessaryReshapePassTest, name)
{
  luci::RemoveUnnecessaryReshapePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveUnnecessaryReshapePass, removed)
{
  ReshapeGraph g;

  g.init({1, 2, 3, 4}, true);

  // confirm graph has Reshape
  auto reshape_node = luci::test::first_node<luci::CircleReshape>(g.g());
  ASSERT_NE(nullptr, reshape_node);
  luci::RemoveUnnecessaryReshapePass pass;
  while (pass.run(g.g()))
    ;

  // check Reshape is removed
  reshape_node = luci::test::first_node<luci::CircleReshape>(g.g());
  ASSERT_EQ(nullptr, reshape_node);
}

TEST(RemoveUnnecessaryReshapePass, not_removed_NEG)
{
  ReshapeGraph g;

  g.init({1, 2, 3, 4}, false);

  // confirm graph has Reshape
  auto reshape_node = luci::test::first_node<luci::CircleReshape>(g.g());
  ASSERT_NE(nullptr, reshape_node);
  luci::RemoveUnnecessaryReshapePass pass;
  while (pass.run(g.g()))
    ;

  // check Reshape is NOT removed
  reshape_node = luci::test::first_node<luci::CircleReshape>(g.g());
  ASSERT_NE(nullptr, reshape_node);
}
