/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_GRAPH_H_
#define _MIR_GRAPH_H_

#include <string>
#include <vector>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <set>

#include "mir/Operation.h"
#include "mir/ops/InputOp.h"
#include "mir/ops/OutputOp.h"

namespace mir
{

class Graph
{
public:
  explicit Graph() = default;

  virtual ~Graph();

  template <typename T, typename... Args> Operation *create(Args &&...args)
  {
    auto op = new T(std::forward<Args>(args)...);
    op->setId(_last_node_id++);
    registerOp(op);
    return op;
  }

  /**
   * @brief Copies `old_op` with new inputs and registers it into graph.
   */
  Operation *copyOpWithInputs(Operation *old_op, const std::vector<Operation::Output *> &inputs)
  {
    assert(inputs.size() == old_op->getNumInputs());
    auto op = old_op->copyWithInputs(inputs);
    op->setId(_last_node_id++);
    registerOp(op);
    return op;
  }

  void accept(IVisitor *visitor);

  /**
   * @brief Returns all graph nodes
   * @return vector containing all graph nodes
   */
  std::unordered_set<Operation *> getNodes() const { return _ops; }

  /**
   * @brief Returns all graph input nodes
   * @returns vector containing all graph input nodes
   */
  std::vector<ops::InputOp *> getInputs() const { return _inputs; }

  /**
   * @brief Returns all graph output nodes
   * @returns vector containing all graph output nodes
   */
  std::vector<ops::OutputOp *> getOutputs() const { return _outputs; }

  /**
   * @brief remove node from graph, along with its links in other nodes
   * @param op node to be removed
   */
  void removeNode(Operation *op);

  /**
   * @brief Subsitude node in graph with another keeping all edges
   * @param op Node to subsitude
   * @param with Node to place instead
   */
  void replaceNode(Operation *op, Operation *with);

private:
  void registerOp(Operation *op);

  std::unordered_set<Operation *> _ops;
  size_t _last_node_id = 0;
  // TODO Change these to unordered_sets.
  std::vector<ops::InputOp *> _inputs;
  std::vector<ops::OutputOp *> _outputs;
};

/**
 * @brief Returns nodes of the graph sorted topologically.
 * @note  Sorting order priority
 * 1) Graph input node (input index order)
 * 2) Constant node (unordered - cannot predict order)
 * 3) Ready node (unordered - cannot predict order)
 */
std::vector<Operation *> getSortedNodes(Graph *graph);

} // namespace mir

#endif //_MIR_GRAPH_H_
