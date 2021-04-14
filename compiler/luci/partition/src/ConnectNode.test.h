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

#ifndef __CONNECT_NODE_TEST_H__
#define __CONNECT_NODE_TEST_H__

#include "ConnectNode.h"

#include <luci/Service/CircleNodeClone.h>
#include <luci/test/TestIOGraph.h>

#include <loco/IR/Graph.h>

#include <memory>
#include <vector>

namespace luci
{
namespace test
{

class TestI2OGraph : public TestIsGraphlet<2>, public TestOGraphlet
{
public:
  TestI2OGraph() = default;

public:
  virtual void init(const ShapeU32 shape_in, const ShapeU32 shape_out)
  {
    TestIsGraphlet<2>::init(g(), {shape_in, shape_in});
    TestOsGraphlet<1>::init(g(), {shape_out});
  }
};

/**
 * @brief ConnectionTestHelper provides common framework for testing
 *        cloned CircleNode connection
 */
class ConnectionTestHelper
{
public:
  ConnectionTestHelper() { _graph_c = loco::make_graph(); }

public:
  void prepare_inputs(TestI2OGraph *i2ograph)
  {
    for (uint32_t i = 0; i < 2; ++i)
    {
      auto *input = _graph_c->nodes()->create<luci::CircleInput>();
      luci::copy_common_attributes(i2ograph->input(i), input);
      _clonectx.emplace(i2ograph->input(i), input);
      _inputs.push_back(input);
    }
  }

  void clone_connect(luci::CircleNode *node, luci::CircleNode *clone)
  {
    _clonectx.emplace(node, clone);

    luci::clone_connect(node, _clonectx);
  }

public:
  loco::Graph *graph_c(void) { return _graph_c.get(); }

  luci::CircleNode *inputs(uint32_t idx) { return _inputs.at(idx); }

protected:
  luci::CloneContext _clonectx;
  std::vector<luci::CircleInput *> _inputs;
  std::unique_ptr<loco::Graph> _graph_c; // new graph for clones
};

} // namespace test
} // namespace luci

#endif // __CONNECT_NODE_TEST_H__
