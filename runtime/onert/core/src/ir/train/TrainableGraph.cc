/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/train/TrainableGraph.h"
#include "util/Utils.h"
#include "util/Set.h"

#include <algorithm>
#include <set>
#include <map>
#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace ir
{
namespace train
{

TrainableGraph::TrainableGraph() : _graph{} {}

TrainableGraph::TrainableGraph(const TrainableGraph &tgraph)
  : _graph{tgraph._graph}, _backward_operands{tgraph._backward_operands}, _losses{tgraph._losses}
{
  tgraph.operations().iterate(
    [&](const onert::ir::OperationIndex &index, const onert::ir::IOperation &op) {
      replaceOperation(index, dynamic_cast<const ITrainableOperation &>(op).clone());
    });
}

TrainableGraph::TrainableGraph(const Graph &graph) : _graph{graph} {}

OperandIndex TrainableGraph::addOperand(const Shape &shape, const TypeInfo &type)
{
  return _graph.addOperand(shape, type);
}

OperandIndex TrainableGraph::addOperand(OperandIndex index, std::unique_ptr<Operand> &&operand)
{
  return _graph.addOperand(index, std::move(operand));
}

OperationIndex TrainableGraph::addOperation(std::unique_ptr<ITrainableOperation> &&operation)
{
  return _graph.addOperation(std::move(operation));
}

OperationIndex TrainableGraph::replaceOperation(OperationIndex index,
                                                std::unique_ptr<ITrainableOperation> &&operation)
{
  return _graph.replaceOperation(index, std::move(operation));
}

OperandIndex TrainableGraph::addBackwardOperand(OperandIndex index,
                                                std::unique_ptr<Operand> &&bwd_operand)
{
  return _backward_operands.push(std::move(bwd_operand), index);
}

IOIndex TrainableGraph::getInputIndex(const std::string &name) const
{
  return _graph.getInputIndex(name);
}

IOIndex TrainableGraph::getOutputIndex(const std::string &name) const
{
  return _graph.getOutputIndex(name);
}

void TrainableGraph::changeShape(const OperandIndex &index, const ir::Shape &new_shape)
{
  _graph.changeShape(index, new_shape);
}

void TrainableGraph::changeBackwardShape(const OperandIndex &index, const ir::Shape &new_shape)
{
  assert(_backward_operands.exist(index));
  _backward_operands.at(index).info().shape(new_shape);
}

void TrainableGraph::addInput(const OperandIndex &ind, const std::string &name)
{
  _graph.addInput(ind, name);
}

void TrainableGraph::addOutput(const OperandIndex &ind, const std::string &name)
{
  _graph.addOutput(ind, name);
}

void TrainableGraph::verify(void) const
{
  _graph.verify();

  operations().iterate([](const onert::ir::OperationIndex &, const onert::ir::IOperation &op) {
    try
    {
      UNUSED_RELEASE(dynamic_cast<const onert::ir::train::ITrainableOperation &>(op));
    }
    catch (const std::bad_cast &)
    {
      throw std::runtime_error("TrainableGraph: " + op.name() + " is not a trainable operation");
    }
  });
}

void TrainableGraph::removeOperand(const OperandIndex &ind) { _graph.removeOperand(ind); }

void TrainableGraph::setLayout(Layout layout) { _graph.setLayout(layout); }

const ITrainableOperation &TrainableGraph::operation(OperationIndex index) const
{
  // NOTE Virtual inherited objects cannot be static_casted.
  return dynamic_cast<const ITrainableOperation &>(_graph.operations().at(index));
}

void TrainableGraph::validateTopologicalOrder(std::vector<ir::OperationIndex> order,
                                              bool is_forward) const
{
  if (!is_forward)
    std::reverse(order.begin(), order.end());

  const std::string order_type = is_forward ? "forward" : "backward";

  std::map<ir::OperationIndex, uint32_t> position;
  for (uint32_t p = 0; p < order.size(); ++p)
  {
    auto index = order[p];
    // TODO: replace this with `std::map::contains` after C++20
    if (position.find(index) != position.end())
      throw std::runtime_error{"Invalid " + order_type + " topological order: duplicate node @" +
                               std::to_string(index.value())};

    position[index] = p;
  }

  operations().iterate([&](const ir::OperationIndex &index, const ir::IOperation &op) {
    if (position.count(index) == 0)
      return;

    uint32_t p = position[index];

    for (const auto &output : op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      const auto &operand = operands().at(output);
      for (const auto &use : operand.getUses())
      {
        if (position.count(use) == 0)
          continue;

        uint32_t q = position[use];
        if (p > q)
          throw std::runtime_error{
            "Invalid " + order_type + " topological order: inversion between @" +
            std::to_string(index.value()) + " and @" + std::to_string(use.value())};
      }
    }
  });
}

void TrainableGraph::validateForwardTopologicalOrder(
  const std::vector<ir::OperationIndex> &order) const
{
  validateTopologicalOrder(order, true);
}

void TrainableGraph::validateBackwardTopologicalOrder(
  const std::vector<ir::OperationIndex> &order) const
{
  validateTopologicalOrder(order, false);
}

std::vector<ir::OperationIndex> TrainableGraph::topolSortOperations() const
{
  auto ret = _graph.topolSortOperations();
  validateForwardTopologicalOrder(ret);

  return ret;
}

std::vector<ir::OperationIndex> TrainableGraph::btopolSortOperations() const
{
  std::vector<ir::OperationIndex> ret;
  util::Set<ir::OperationIndex> unvisited;
  ir::OperationIndex loss_idx;
  operations().iterate([&](const ir::OperationIndex &index, const ir::IOperation &op) {
    unvisited.add(index);
    if (op.opcode() == ir::OpCode::Loss)
    {
      assert(!loss_idx.valid()); // Should be only one loss
      loss_idx = index;
    }
  });

  std::function<void(const ir::OperationIndex &, const ir::IOperation &)> dfs =
    [&](const ir::OperationIndex &index, const ir::IOperation &op) -> void {
    if (!unvisited.contains(index))
      return;
    unvisited.remove(index);

    for (const auto &input : op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      const auto &operand = operands().at(input);
      const auto &def = operand.getDef();
      if (!def.valid())
        continue;
      dfs(def, operations().at(def));
    }

    ret.push_back(index);
  };

  dfs(loss_idx, operations().at(loss_idx));
  std::reverse(ret.begin(), ret.end());
  validateBackwardTopologicalOrder(ret);

  return ret;
}

std::vector<ir::OperationIndex>
TrainableGraph::truncateBackwardOrder(std::vector<ir::OperationIndex> backward_order) const
{
  auto forward_order = backward_order;
  std::reverse(forward_order.begin(), forward_order.end());

  std::set<ir::OperationIndex> alive;

  for (const auto &index : forward_order)
  {
    const auto &op = operations().at(index);
    const auto &trainable_op = dynamic_cast<const ITrainableOperation &>(op);

    if (trainable_op.hasTrainableParameter())
      alive.insert(index);

    // TODO: replace this with `std::set::contains` after C++20
    if (alive.find(index) != alive.end())
    {
      for (const auto &output : op.getOutputs())
      {
        const auto &operand = operands().at(output);
        for (const auto &use : operand.getUses())
        {
          alive.insert(use);
        }
      }
    }
  }

  backward_order.erase(std::remove_if(backward_order.begin(), backward_order.end(),
                                      [&](const auto &index) {
                                        // TODO: replace this with `std::set::contains` after C++20
                                        return alive.find(index) == alive.end();
                                      }),
                       backward_order.end());

  return backward_order;
}

void TrainableGraph::addLoss(const OperandIndex &loss_ind, const IOIndex &pred_ioind)
{
  _losses.emplace(pred_ioind, loss_ind);
}

OperandIndex TrainableGraph::getLossIndex(const IOIndex &pred_ioind) const
{
  auto itr = _losses.find(pred_ioind);
  return (itr == _losses.end()) ? OperandIndex{} : itr->second;
}

} // namespace train
} // namespace ir
} // namespace onert
