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

#include "luci/Pass/RemoveUnnecessaryCastPass.h"
#include "helpers/CreateCircleConst.h"

#include <luci/IR/CircleNodes.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

template <typename T>
luci::CircleConst *const_node_of_dtype(loco::Graph *g, const loco::DataType dtype,
                                       const std::vector<uint32_t> &shape, T value)
{
  switch (dtype)
  {
    case loco::DataType::S32:
      return luci::create_const_node(g, dtype, shape, static_cast<int32_t>(value));
    case loco::DataType::FLOAT32:
      return luci::create_const_node(g, dtype, shape, static_cast<float>(value));
    default:
      throw std::runtime_error("Unsupported dtype!");
  }
}

/**
 *  Graph for this test
 *
 *  BEFORE
 *
 *           |
 *      [CircleAdd]
 *           |
 *      [CircleCast]
 *           |
 *      [CircleAdd]
 *           |
 *
 *  AFTER
 *
 *           |
 *      [CircleAdd]
 *           |           [CircleCast]
 *      [CircleAdd]
 *           |
 *
 */
class CastGraphlet
{
public:
  void init(loco::Graph *g, loco::DataType in_type, loco::DataType out_type)
  {
    _const_a = const_node_of_dtype(g, in_type, {1}, 1);

    _add_a = g->nodes()->create<luci::CircleAdd>();
    // _add_a->x(input_of_the_net);
    _add_a->y(_const_a);
    _add_a->dtype(in_type);
    _add_a->shape({1});
    _add_a->name("add_a");

    _cast = g->nodes()->create<luci::CircleCast>();
    _cast->in_data_type(in_type);
    _cast->out_data_type(out_type);
    _cast->x(_add_a);

    _const_a = const_node_of_dtype(g, out_type, {1}, 2);

    _add_b = g->nodes()->create<luci::CircleAdd>();
    _add_b->x(_cast);
    _add_b->y(_const_b);
    _add_b->dtype(out_type);
    _add_b->shape({1});
    _add_b->name("add_b");
  }

protected:
  luci::CircleCast *_cast = nullptr;
  luci::CircleAdd *_add_a = nullptr;
  luci::CircleConst *_const_a = nullptr;
  luci::CircleAdd *_add_b = nullptr;
  luci::CircleConst *_const_b = nullptr;
};

class RemoveUnnecessaryCastTestGraph : public TestIOGraph, public CastGraphlet
{
public:
  void init(loco::DataType in_type, loco::DataType out_type)
  {
    TestIOGraph::init({1}, {1});
    CastGraphlet::init(g(), in_type, out_type);

    _add_a->x(input());

    output()->from(_add_b);
  }
};

class RemoveUnnecessaryCastPassTest : public ::testing::Test
{
public:
  RemoveUnnecessaryCastTestGraph g;
  luci::RemoveUnnecessaryCastPass pass;
};

} // namespace

TEST_F(RemoveUnnecessaryCastPassTest, cast_remove)
{
  g.init(loco::DataType::FLOAT32 /* in_type */, loco::DataType::FLOAT32 /* out_type */);

  EXPECT_EQ(true, pass.run(g.g()));

  auto last_add = dynamic_cast<luci::CircleAdd *>(g.output()->from());
  EXPECT_NE(nullptr, last_add);
  // Check if the cast was removed:
  auto first_add = dynamic_cast<luci::CircleAdd *>(last_add->x());
  EXPECT_NE(nullptr, first_add);
}

TEST_F(RemoveUnnecessaryCastPassTest, different_data_types_NEG)
{
  g.init(loco::DataType::FLOAT32 /* in_type */, loco::DataType::S32 /* out_type */);

  EXPECT_EQ(false, pass.run(g.g()));
}
