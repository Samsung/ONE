/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveDuplicateConstPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/test/TestIOGraph.h>
#include <gtest/gtest.h>

namespace
{
using namespace luci::test;

class DuplicateConstsGraphlet
{
public:
  DuplicateConstsGraphlet() = default;

public:
  void init(loco::Graph *g, bool is_duplicate)
  {
    _reshape_shape = g->nodes()->create<luci::CircleConst>();
    _reshape_shape->rank(1);
    _reshape_shape->dim(0).set(1);
    _reshape_shape->shape_status(luci::ShapeStatus::VALID);
    _reshape_shape->dtype(loco::DataType::S32);

    _reshape_shape->size<loco::DataType::S32>(1);
    _reshape_shape->at<loco::DataType::S32>(0) = 5;
    _reshape_shape->name("reshape_shape_1");

    _reshape_shape_duplicate = g->nodes()->create<luci::CircleConst>();
    _reshape_shape_duplicate->rank(1);
    _reshape_shape_duplicate->dim(0).set(1);
    _reshape_shape_duplicate->shape_status(luci::ShapeStatus::VALID);
    _reshape_shape_duplicate->dtype(loco::DataType::S32);
    if (is_duplicate)
    {
      _reshape_shape_duplicate->size<loco::DataType::S32>(1);
      _reshape_shape_duplicate->at<loco::DataType::S32>(0) = 5;
    }
    else
    {
      _reshape_shape_duplicate->size<loco::DataType::S32>(2);
      _reshape_shape_duplicate->at<loco::DataType::S32>(0) = 1;
      _reshape_shape_duplicate->at<loco::DataType::S32>(1) = 5;
    }
    _reshape_shape_duplicate->name("reshape_shape_2");

    _reshape_f = g->nodes()->create<luci::CircleReshape>();
    _reshape_f->newShape()->rank(1);
    _reshape_f->newShape()->dim(0) = 5;
    _reshape_f->name("reshape_f");

    _reshape_s = g->nodes()->create<luci::CircleReshape>();
    if (is_duplicate)
    {
      _reshape_s->newShape()->rank(1);
      _reshape_s->newShape()->dim(0) = 5;
    }
    else
    {
      _reshape_s->newShape()->rank(2);
      _reshape_s->newShape()->dim(0) = 1;
      _reshape_s->newShape()->dim(1) = 5;
    }
    _reshape_s->name("reshape_s");
  }

protected:
  luci::CircleReshape *_reshape_f = nullptr;
  luci::CircleReshape *_reshape_s = nullptr;
  luci::CircleConst *_reshape_shape = nullptr;
  luci::CircleConst *_reshape_shape_duplicate = nullptr;
};

class DuplicateConstsGraph : public TestIOGraph, public DuplicateConstsGraphlet
{
public:
  DuplicateConstsGraph() = default;

public:
  void init(const ShapeU32 in_shape, const ShapeU32 out_shape, bool is_duplicate)
  {
    TestIOGraph::init(in_shape, out_shape);

    DuplicateConstsGraphlet::init(g(), is_duplicate);

    // connect graph
    _reshape_f->tensor(input());
    _reshape_f->shape(_reshape_shape);

    _reshape_s->tensor(_reshape_f);
    _reshape_s->shape(_reshape_shape_duplicate);

    output()->from(_reshape_s);
  }
};
} // namespace

TEST(RemoveDuplicateConstPass, name)
{
  luci::RemoveDuplicateConstPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(RemoveDuplicateConstPass, remove_duplicate)
{
  DuplicateConstsGraph g;
  g.init({1, 5}, {5}, true);

  luci::RemoveDuplicateConstPass pass;
  while (pass.run(g.g()))
    ;

  uint32_t const_num = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    auto target_node = dynamic_cast<luci::CircleConst *>(node);
    if (target_node != nullptr)
      const_num++;
  }

  ASSERT_EQ(const_num, 1);
}

TEST(RemoveDuplicateConstPass, remove_duplicate_NEG)
{
  DuplicateConstsGraph g;
  g.init({1, 5}, {1, 5}, false);

  luci::RemoveDuplicateConstPass pass;
  while (pass.run(g.g()))
    ;

  uint32_t const_num = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g.g())))
  {
    auto target_node = dynamic_cast<luci::CircleConst *>(node);
    if (target_node != nullptr)
      const_num++;
  }

  ASSERT_EQ(const_num, 2);
}
