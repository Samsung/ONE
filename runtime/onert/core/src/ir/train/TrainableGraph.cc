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

std::vector<ir::OperationIndex> TrainableGraph::topolSortOperations() const
{
  return _graph.topolSortOperations();
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
    ret.push_back(index);

    for (const auto &input : op.getInputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      const auto &operand = operands().at(input);
      const auto &def = operand.getDef();
      if (!def.valid())
        return;
      dfs(def, operations().at(def));
    }
  };

  dfs(loss_idx, operations().at(loss_idx));

  return ret;
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
