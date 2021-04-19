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

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace luci
{
namespace test
{

template <unsigned N> class TestIsOGraph : public TestIsGraphlet<N>, public TestOGraphlet
{
public:
  TestIsOGraph() = default;

public:
  virtual void init(const std::initializer_list<ShapeU32> shape_in, const ShapeU32 shape_out)
  {
    if (shape_in.size() != N)
      throw std::runtime_error("Failed to init TestIsOGraph");

    TestIsGraphlet<N>::init(TestIsGraphlet<N>::g(), shape_in);
    TestOGraphlet::init(TestIsGraphlet<N>::g(), shape_out);
  }
};

template <class T> class NodeGraphletT
{
public:
  virtual void init(loco::Graph *g)
  {
    _node = g->nodes()->create<T>();
    _node->dtype(loco::DataType::S32);
    _node->name("node");
  }

  T *node(void) const { return _node; }

protected:
  T *_node{nullptr};
};

template <class T> class NodeIsGraphletT
{
public:
  virtual void init(loco::Graph *g, uint32_t n)
  {
    _node = g->nodes()->create<T>(n);
    _node->dtype(loco::DataType::S32);
    _node->name("node");
  }

  T *node(void) const { return _node; }

protected:
  T *_node{nullptr};
};

/**
 * @brief ConnectionTestHelper provides common framework for testing
 *        cloned CircleNode connection
 */
class ConnectionTestHelper
{
public:
  ConnectionTestHelper() { _graph_clone = loco::make_graph(); }

public:
  template <unsigned N> void prepare_inputs(TestIsOGraph<N> *isograph)
  {
    assert(N == isograph->num_inputs());

    for (uint32_t i = 0; i < N; ++i)
    {
      auto *input = _graph_clone->nodes()->create<luci::CircleInput>();
      luci::copy_common_attributes(isograph->input(i), input);
      _clonectx.emplace(isograph->input(i), input);
      _inputs.push_back(input);
    }
  }

  /**
   * @note prepare_inputs_miss is for negative testing
   */
  template <unsigned N> void prepare_inputs_miss(TestIsOGraph<N> *isograph)
  {
    assert(N == isograph->num_inputs());

    for (uint32_t i = 0; i < N; ++i)
    {
      auto *input = _graph_clone->nodes()->create<luci::CircleInput>();
      luci::copy_common_attributes(isograph->input(i), input);
      if (i != 0)
        _clonectx.emplace(isograph->input(i), input);
      _inputs.push_back(input);
    }
  }

  void clone_connect(luci::CircleNode *node, luci::CircleNode *clone)
  {
    _clonectx.emplace(node, clone);

    luci::clone_connect(node, _clonectx);
  }

public:
  loco::Graph *graph_clone(void) { return _graph_clone.get(); }

  luci::CircleNode *inputs(uint32_t idx) { return _inputs.at(idx); }

protected:
  luci::CloneContext _clonectx;
  std::vector<luci::CircleInput *> _inputs;
  std::unique_ptr<loco::Graph> _graph_clone; // graph for clones
};

} // namespace test
} // namespace luci

#endif // __CONNECT_NODE_TEST_H__
