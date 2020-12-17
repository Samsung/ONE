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

#include "mir/Graph.h"

#include <algorithm>
#include <deque>
#include <unordered_map>

namespace mir
{

/**
 * @brief replace all usages of operation `op` with node `with`
 * (i.e. all references in previous/next nodes )
 * @param op the operation to replace
 * @param with the operation to use as a replacement
 */
static void replaceUsages(Operation *op, Operation *with)
{
  assert(op->getNumOutputs() == with->getNumOutputs());
  for (std::size_t i = 0; i < op->getNumOutputs(); ++i)
  {
    Operation::Output *output = op->getOutput(i);
    output->replaceAllUsesWith(with->getOutput(i));
  }
}

std::vector<Operation *> getSortedNodes(Graph *graph)
{
  std::deque<Operation *> ready_nodes;
  std::unordered_map<Operation *, std::size_t> num_visited_input_edges;

  for (Operation *op : graph->getNodes())
  {
    if (op->getNumInputs() == 0)
    {
      ready_nodes.push_back(op);
    }
  }

  std::vector<Operation *> sorted_nodes;
  while (!ready_nodes.empty())
  {
    Operation *src_node = ready_nodes.front();
    ready_nodes.pop_front();
    sorted_nodes.push_back(src_node);
    for (Operation::Output &output : src_node->getOutputs())
    {
      for (const auto use : output.getUses())
      {
        Operation *dst_node = use.getNode();
        if (++num_visited_input_edges[dst_node] == dst_node->getNumInputs())
        {
          ready_nodes.push_back(dst_node);
        }
      }
    }
  }

  return sorted_nodes;
}

void Graph::accept(IVisitor *visitor)
{
  for (Operation *node : getSortedNodes(this))
  {
    node->accept(visitor);
  }
}

Graph::~Graph()
{
  for (auto &node : _ops)
  {
    delete node;
  }
}

void Graph::registerOp(Operation *op)
{
  _ops.emplace(op);

  if (auto *input_op = dynamic_cast<ops::InputOp *>(op))
    _inputs.emplace_back(input_op);

  if (auto *output_op = dynamic_cast<ops::OutputOp *>(op))
    _outputs.emplace_back(output_op);
}

void Graph::replaceNode(Operation *op, Operation *with)
{
  replaceUsages(op, with);
  removeNode(op);
}

void Graph::removeNode(Operation *op)
{
#ifndef NDEBUG
  for (const auto &output : op->getOutputs())
  {
    assert(output.getUses().empty() && "Trying to remove a node that has uses.");
  }
#endif

  for (std::size_t i = 0; i < op->getNumInputs(); ++i)
  {
    op->getInput(i)->removeUse(Operation::Use(op, i));
  }

  if (op->getType() == Operation::Type::input)
    _inputs.erase(
      std::remove(_inputs.begin(), _inputs.end(), op)); // NOLINT(bugprone-inaccurate-erase)

  if (op->getType() == Operation::Type::output)
    _outputs.erase(
      std::remove(_outputs.begin(), _outputs.end(), op)); // NOLINT(bugprone-inaccurate-erase)

  _ops.erase(op);
  delete op;
}

} // namespace mir
