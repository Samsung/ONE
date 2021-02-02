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

#ifndef NNCC_SUBGRAPH_CONTEXT_H
#define NNCC_SUBGRAPH_CONTEXT_H

#include "luci/IR/CircleNodeDecl.h"

#include "Halide.h"

#include <unordered_map>
#include <vector>
#include <utility>
#include <string>

namespace luci_codegen
{

/**
 * @brief This class is responsible for holding subgraph nodes and corresponding halide entities
 */
class SubgraphContext
{
public:
  SubgraphContext(std::string name = ""): _name(std::move(name))
  {
#ifndef NDEBUG
    _constructed = false;
#endif
  }

  template<typename Cont = std::vector<luci::CircleNode *>>
  SubgraphContext(std::string name, Cont &&nodes): _name(std::move(name)), _nodes(std::forward<Cont>(nodes))
  {
#ifndef NDEBUG
    _constructed = false;
#endif
  }

  SubgraphContext(const SubgraphContext &sub) = default;

  SubgraphContext(SubgraphContext &&sub) = default;

  // Construction methods

  /**
   * @brief construction method, adds node to subgraph
   * @param node node to add in subgraph
   */
  void add_node(luci::CircleNode *node)
  {
#ifndef NDEBUG
    assert(!_constructed);
    assert(_nodes.empty() || node->graph() == get_graph());
#endif
    _nodes.push_back(node);
  }

  /**
   * @brief gathers inputs and outputs, generates functions for operators
   * after this call no construction methods are allowed
   */
  void finish_construction();

  std::string get_name() const
  {
    return _name;
  }

  loco::Graph *get_graph() const
  {
    assert(!_nodes.empty() && "graph is not filled add at least one node");
    return _nodes[0]->graph();
  }

  /**
   * @return nodes in subgraph
   */
  const std::vector<luci::CircleNode *> get_nodes() const
  {
    return _nodes;
  }

  /**
   * @param node target node, it should belong to graph or graph inputs
   * @return function created for given node
   */
  Halide::Func get_func(loco::Node *node) const;

  /**
   * @param node target node
   * @return true if node belongs to subgraph, false otherwise
   */
  bool contains(luci::CircleNode *node) const
  {
    return _generated_funcs.count(node);
  }

  /**
   * @return vector of inputs (needed for halide code generation)
   */
  const std::vector<std::pair<luci::CircleNode *, Halide::ImageParam>> &get_inputs() const
  {
    assert(_constructed);
    return _inputs;
  }

  /**
   * @return vector of outputs (needed to generate outputs of subgraph)
   */
  const std::vector<std::pair<luci::CircleNode *, Halide::Func>> &get_outputs() const
  {
    assert(_constructed);
    return _outputs;
  }

private:
#ifndef NDEBUG
  bool _constructed;
#endif
  std::string _name;
  std::vector<luci::CircleNode *> _nodes;
  std::unordered_map<loco::Node *, Halide::Func> _generated_funcs;
  std::vector<std::pair<luci::CircleNode *, Halide::ImageParam>> _inputs;
  std::vector<std::pair<luci::CircleNode *, Halide::Func>> _outputs;
};

}

#endif //NNCC_SUBGRAPH_CONTEXT_H
