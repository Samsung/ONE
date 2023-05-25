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

#include "util/Set.h"

namespace onert
{
namespace ir
{
namespace train
{

TrainableGraph::TrainableGraph(const Graph &graph) : _graph{graph}, _operations{} {}

TrainableGraph::~TrainableGraph(void) = default;

OperandIndex TrainableGraph::addOperand(const Shape &shape, const TypeInfo &type)
{
  return _graph.addOperand(shape, type);
}

OperandIndex TrainableGraph::addOperand(OperandIndex index, std::unique_ptr<Operand> &&operand)
{
  return _graph.addOperand(index, std::move(operand));
}

OperationIndex
TrainableGraph::addTrainableOperation(OperationIndex index,
                                      std::unique_ptr<ITrainableOperation> &&operation)
{
  const auto &inputs = operation->getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
  const auto &outputs = operation->getOutputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
  for (auto input : inputs)
    if (!operands().exist(input))
      return OperationIndex{};
  for (auto output : outputs)
    if (!operands().exist(output))
      return OperationIndex{};

  auto ind_gen = _operations.push(std::move(operation), index);
  if (ind_gen.valid())
  {
    assert(ind_gen == index);
  }

  // TODO Link operands to trainable operation forwarding and backwarding

  return index;
}

OperationIndex TrainableGraph::addOperation(std::unique_ptr<Operation> &&operation)
{
  return _graph.addOperation(std::move(operation));
}

OperationIndex TrainableGraph::addOperation(OperationIndex index,
                                            std::unique_ptr<Operation> &&operation)
{
  return _graph.addOperation(index, std::move(operation));
}

void TrainableGraph::setOperandValue(const OperandIndex &ind, std::shared_ptr<Data> data)
{
  _graph.setOperandValue(ind, data);
}

IOIndex TrainableGraph::getInputIndex(const std::string &name) const
{
  return _graph.getInputIndex(name);
}

IOIndex TrainableGraph::getOutputIndex(const std::string &name) const
{
  return _graph.getOutputIndex(name);
}

void TrainableGraph::verify(void)
{
  _graph.verify();
  // TODO Verify TrainableGraph
}

} // namespace train
} // namespace ir
} // namespace onert
