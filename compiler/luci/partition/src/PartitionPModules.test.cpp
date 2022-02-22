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

#include "PartitionPModules.h"
#include "PartitionPGroups.h"

#include <luci/test/TestIOGraph.h>

#include <luci/IR/Nodes/CircleSqrt.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class SqrtGraphlet
{
public:
  SqrtGraphlet() = default;

public:
  void init(loco::Graph *g, const ShapeU32 input_shape)
  {
    _sqrt = g->nodes()->create<luci::CircleSqrt>();
    _sqrt->dtype(loco::DataType::S32);
    _sqrt->name("sqrt");
  }

protected:
  luci::CircleSqrt *_sqrt = nullptr;
};

class SqrtGraph : public TestIOGraph, public SqrtGraphlet
{
public:
  SqrtGraph() = default;

public:
  void init(const ShapeU32 shape)
  {
    TestIOGraph::init(shape, shape);
    SqrtGraphlet::init(g(), shape);

    _sqrt->x(input());

    output()->from(_sqrt);
  }
};

} // namespace

TEST(PartitionPModulesTest, simple_convert)
{
  luci::Module module;

  SqrtGraph g;
  g.init({3, 3});
  g.transfer_to(&module);

  luci::PartitionTable pt;
  pt.default_group = "A";
  pt.comply = luci::PartitionTable::COMPLY::OPCODE;

  auto pgs = produce_pgroups(&module, pt);
  auto pms = produce_pmodules(pgs.get());

  ASSERT_EQ(1, pms.pmodules.size());
}
