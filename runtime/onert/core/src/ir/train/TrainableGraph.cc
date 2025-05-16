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

#include "ir/OperandIndexMap.h"
#include "UseDefGenerator.h"
#include "util/Set.h"
#include "../verifier/Verifier.h"

#include <algorithm>
#include <set>
#include <map>
#include <misc/polymorphic_downcast.h>

namespace
{

using namespace onert;
using namespace onert::ir;
using namespace onert::ir::train;

void disableUnusedBackwardNodes(const UseDefChains &training_usedefs, TrainableGraph &tgraph)
{
  // Disable backward nodes that will be unused
  const auto border = tgraph.btopolSortOperations();
  for (const auto &op_index : border)
  {
    const auto &node = tgraph.operations().at(op_index);
    const auto &candidates =
      (node.getInputs() + node.getOutputs()) | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
    const bool is_backward_op_used =
      std::any_of(candidates.begin(), candidates.end(), [&](const OperandIndex &operand) {
        const auto training_op_index = TrainingOperationIndex{op_index, false};
        const auto forwarding_index = TrainingOperandIndex{operand, true};
        const auto &forwarding_uses = training_usedefs.at(forwarding_index).getTrainingUses();
        const auto backwarding_index = TrainingOperandIndex{operand, false};
        const auto &backwarding_uses = training_usedefs.at(backwarding_index).getTrainingUses();
        return forwarding_uses.find(training_op_index) != forwarding_uses.end() ||
               backwarding_uses.find(training_op_index) != backwarding_uses.end();
      });

    // NOTE Backward op does not define any incoming operand in backwarding
    const auto &inputs = node.getUsedInputSet();
    const bool is_backward_op_def =
      std::any_of(inputs.begin(), inputs.end(), [&](const OperandIndex &input) {
        const auto training_op_index = TrainingOperationIndex{op_index, false};
        const auto outcoming_index = TrainingOperandIndex{input, false};
        const auto &backwarding_defs = training_usedefs.at(outcoming_index).getTrainingUses();
        return backwarding_defs.find(training_op_index) != backwarding_defs.end();
      });

    if (is_backward_op_used || is_backward_op_def)
      tgraph.enableBackward(op_index);
    else
      tgraph.disableBackward(op_index);
  }
}

} // namespace

namespace onert::ir::train
{

TrainableGraph::TrainableGraph() : _graph{} {}

TrainableGraph::TrainableGraph(const TrainableGraph &tgraph)
  : _graph{tgraph._graph}, _backward_operands{tgraph._backward_operands},
    _training_defuses{tgraph._training_defuses}, _losses{tgraph._losses}
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
      [[maybe_unused]] const auto &casted_op =
        dynamic_cast<const onert::ir::train::ITrainableOperation &>(op);
    }
    catch (const std::bad_cast &)
    {
      throw std::runtime_error("TrainableGraph: " + op.name() + " is not a trainable operation");
    }
  });

  verifyTrainingUseDefs();
}

void TrainableGraph::removeOperand(const OperandIndex &ind) { _graph.removeOperand(ind); }

const ITrainableOperation &TrainableGraph::operation(OperationIndex index) const
{
  // NOTE Virtual inherited objects cannot be static_casted.
  return dynamic_cast<const ITrainableOperation &>(_graph.operations().at(index));
}

void TrainableGraph::enableBackward(const OperationIndex &index)
{
  auto op = dynamic_cast<ir::train::ITrainableOperation *>(&_graph.operations().at(index));
  assert(op);
  op->enableBackward();
}

void TrainableGraph::disableBackward(const OperationIndex &index)
{
  auto &op = dynamic_cast<ir::train::ITrainableOperation &>(_graph.operations().at(index));
  op.disableBackward();
}

void TrainableGraph::setTrainingUseDefs(const UseDefChains &training_defuses)
{
  _training_defuses.clear();
  // TODO Replace this loop with `std::unordered_map::insert_range` since C++23
  for (const auto &[training_index, usedef] : training_defuses)
  {
    _training_defuses.emplace(training_index, usedef);
  }
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

    for (const auto &output : op.getUsedOutputSet())
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

void TrainableGraph::verifyTrainingUseDefs() const
{
  if (!verifier::DAGChecker().verify(_training_defuses))
    throw std::runtime_error{"The training def-uses is cyclic."};
  assert(verifier::EdgeChecker().verify(_training_defuses));
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

    for (const auto &input : op.getUsedInputSet())
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

std::vector<ir::OperationIndex> TrainableGraph::essentialBackwardOrder() const
{
  auto backward_order = btopolSortOperations();
  // get rid of all nodes not reachable from a node with trainable parameters
  backward_order = truncateBackwardOrder(backward_order, [&](const OperationIndex &index) {
    return operation(index).isRequiredForBackward();
  });

  return truncateBackwardOrder(backward_order);
}

std::vector<ir::OperationIndex> TrainableGraph::truncateBackwardOrder(
  std::vector<ir::OperationIndex> backward_order,
  std::function<bool(const ir::OperationIndex &)> alive_cond) const
{
  auto forward_order = backward_order;
  std::reverse(forward_order.begin(), forward_order.end());
  std::set<ir::OperationIndex> alive;

  for (const auto &index : forward_order)
  {
    if (alive_cond(index))
      alive.insert(index);

    // TODO: replace this with `std::set::contains` after C++20
    if (alive.find(index) != alive.end())
    {
      const auto &op = operations().at(index);
      for (const auto &output : op.getOutputs())
      {
        const auto &operand = operands().at(output);
        for (const auto &use : operand.getUses())
          alive.insert(use);
      }
    }
  }

  // TODO: replace this with `std::erase_if(std::vector)` after C++20
  backward_order.erase(
    std::remove_if(backward_order.begin(), backward_order.end(),
                   [&](const auto &index) { return alive.find(index) == alive.end(); }),
    backward_order.end());
  return backward_order;
}

std::vector<ir::OperationIndex>
TrainableGraph::truncateBackwardOrder(const std::vector<ir::OperationIndex> &backward_order) const
{
  return truncateBackwardOrder(backward_order, [&](const ir::OperationIndex &index) {
    const auto &trainable_op = operation(index);

    return trainable_op.hasTrainableParameter();
  });
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

void TrainableGraph::updateGraphDependency()
{
  _graph.verify();

  // Initialize training usedefs
  setTrainingUseDefs(UseDefGenerator{*this}());

  disableUnusedBackwardNodes(_training_defuses, *this);

  verifyTrainingUseDefs();
}

} // namespace onert::ir::train
