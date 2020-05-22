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

#ifndef __TEST_GRAPH_H__
#define __TEST_GRAPH_H__

#include <luci/IR/CircleNodes.h>

#include <loco.h>

#include <cassert>
#include <memory>

// TODO Change all Canonical nodes to Circle nodes

namespace luci
{
namespace test
{

class TestGraph
{
public:
  std::unique_ptr<loco::Graph> g;
  luci::CircleInput *input_node = nullptr;
  luci::CircleOutput *output_node = nullptr;

  TestGraph() // creates Pull and Push
  {
    g = loco::make_graph();

    input_node = g->nodes()->create<luci::CircleInput>();

    output_node = g->nodes()->create<luci::CircleOutput>();

    auto input = g->inputs()->create();
    {
      input->name("input");
      luci::link(input, input_node);
    }
    auto output = g->outputs()->create();
    {
      output->name("output");
      luci::link(output, output_node);
    }

    _next_input = input_node;
  }

  loco::Graph *graph() { return g.get(); }

  /// @brief Creates node with NO arg and appends it to graph
  template <class T> T *append()
  {
    auto node = g->nodes()->create<T>();
    _next_input = node;

    return node;
  }

  /// @brief Creates op T (arity=1) with arg1 as an input and appends it to graph
  template <class T> T *append(luci::CircleNode *arg1)
  {
    auto node = g->nodes()->create<T>();
    setInput(node, arg1);
    _next_input = node;

    return node;
  }

  /// @brief Creates op T (arity=2) with arg1, arg2 as inputs and appends it to graph
  template <class T> T *append(luci::CircleNode *arg1, luci::CircleNode *arg2)
  {
    auto node = g->nodes()->create<T>();
    setInput(node, arg1, arg2);
    _next_input = node;

    return node;
  }

  /// @brief Creates op T (arity=3) with arg1, arg2, arg3 as inputs and appends it to graph
  template <class T>
  T *append(luci::CircleNode *arg1, luci::CircleNode *arg2, luci::CircleNode *arg3)
  {
    auto node = g->nodes()->create<T>();
    setInput(node, arg1, arg2, arg3);
    _next_input = node;

    return node;
  }

  // output will get the last appended node
  void complete() { output_node->from(_next_input); }

  void complete(luci::CircleNode *last_node) { output_node->from(last_node); }

private:
  // arity 1
  void setInput(luci::CircleNode *, luci::CircleNode *) { assert(false && "NYI"); };

  void setInput(luci::CircleAveragePool2D *node, luci::CircleNode *input) { node->value(input); };
  void setInput(luci::CircleRelu *node, luci::CircleNode *input) { node->features(input); };
  void setInput(luci::CircleSqueeze *node, luci::CircleNode *input) { node->input(input); };

  void setInput(luci::CircleGatherNd *node, luci::CircleNode *params, luci::CircleNode *indices)
  {
    node->params(params);
    node->indices(indices);
  };

  // arity 2
  void setInput(luci::CircleNode *, luci::CircleNode *, luci::CircleNode *)
  {
    assert(false && "NYI");
  };

  void setInput(luci::CircleExpandDims *node, luci::CircleNode *arg1, luci::CircleNode *arg2)
  {
    node->input(arg1);
    node->axis(arg2);
  };

  void setInput(luci::CircleTranspose *node, luci::CircleNode *arg1, luci::CircleNode *arg2)
  {
    node->a(arg1);
    node->perm(arg2);
  };

  void setInput(luci::CircleResizeNearestNeighbor *node, luci::CircleNode *input,
                luci::CircleNode *size)
  {
    node->input(input);
    node->size(size);
  };

  // arity 3
  void setInput(luci::CircleNode *, luci::CircleNode *, luci::CircleNode *, luci::CircleNode *)
  {
    assert(false && "NYI");
  };

private:
  loco::Node *_next_input;
};

enum class ExampleGraphType
{
  CircleTranspose,
};

template <ExampleGraphType T> class ExampleGraph;

/**
 * @brief Class to create the following:
 *
 *     CircleInput -- CircleTranspose -- CircleOutput
 */
template <> class ExampleGraph<ExampleGraphType::CircleTranspose> : public TestGraph
{
public:
  luci::CircleConst *const_perm = nullptr;
  luci::CircleTranspose *transpose_node = nullptr;

public:
  ExampleGraph()
  {
    const_perm = append<luci::CircleConst>();
    transpose_node = append<luci::CircleTranspose>(input_node, const_perm);
    complete(transpose_node);
  }
};

} // namespace test
} // namespace luci

namespace luci
{
namespace test
{

/// @brief This will set GraphInput shape from CircleInput shape
void graph_input_shape(luci::CircleInput *input);

/// @brief This will set GraphOutput shape from CircleOutput shape
void graph_output_shape(luci::CircleOutput *output);

/// @brief This will set GraphInput dtype from CircleInput dtype
void graph_input_dtype(luci::CircleInput *input);

/// @brief This will set GraphOutput dtype from CircleOutput dtype
void graph_output_dtype(luci::CircleOutput *output);

} // namespace test
} // namespace luci

#endif // __TEST_GRAPH_H__
