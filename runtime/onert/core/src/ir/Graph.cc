/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/Graph.h"

#include "OperationValidator.h"
#include "verifier/Verifier.h"

#include "util/Set.h"

namespace onert
{
namespace ir
{

Graph::Graph() = default;

Graph::Graph(const Graph &) = default;

Graph::~Graph(void) = default;

OperandIndex Graph::addOperand(const Shape &shape, const TypeInfo &type)
{
  return _operands.emplace(shape, type);
}

OperandIndex Graph::addOperand(OperandIndex index, std::unique_ptr<Operand> &&operand)
{
  return _operands.push(std::move(operand), index);
}

bool Graph::checkOperandsForOperation(const IOperation &operation)
{
  auto inputs = operation.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
  auto outputs = operation.getOutputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
  for (auto input : inputs)
    if (!operands().exist(input))
      return false;
  for (auto input : outputs)
    if (!operands().exist(input))
      return false;
  return true;
}

void Graph::linkOperandToOperation(OperationIndex index, const IOperation &operation)
{
  auto inputs = operation.getInputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;
  auto outputs = operation.getOutputs() | ir::Remove::UNDEFINED | ir::Remove::DUPLICATED;

  for (auto input : inputs)
    operands().at(input).insertUse(index);
  for (auto output : outputs)
    operands().at(output).setDef(index);
}

OperationIndex Graph::addOperation(std::unique_ptr<IOperation> &&operation)
{
  const IOperation &op_ref = *operation;
  if (!checkOperandsForOperation(op_ref))
    return OperationIndex{};
  auto ind = _operations.push(std::move(operation));
  if (ind.valid())
    linkOperandToOperation(ind, op_ref);
  return ind;
}

OperationIndex Graph::addOperation(OperationIndex index, std::unique_ptr<IOperation> &&operation)
{
  const IOperation &op_ref = *operation;
  if (!checkOperandsForOperation(op_ref))
    return OperationIndex{};
  auto ind_gen = _operations.push(std::move(operation), index);
  if (ind_gen.valid())
  {
    assert(ind_gen == index);
    linkOperandToOperation(index, op_ref);
  }
  return index;
}

OperationIndex Graph::replaceOperation(OperationIndex index,
                                       std::unique_ptr<IOperation> &&operation)
{
  const IOperation &op_ref = *operation;
  if (!checkOperandsForOperation(op_ref) || !_operations.exist(index))
    return OperationIndex{};

  return _operations.push(std::move(operation), index);
}

void Graph::setOperandValue(const OperandIndex &ind, std::shared_ptr<Data> data)
{
  assert(_operands.exist(ind));
  _operands.at(ind).data(std::move(data));
}

void Graph::addInput(const OperandIndex &ind, const std::string &name)
{
  if (!name.empty())
    _io_info.name_to_input.emplace(name, IOIndex{_io_info.inputs.size()});
  _io_info.inputs.append(ind);
}

void Graph::addOutput(const OperandIndex &ind, const std::string &name)
{
  if (!name.empty())
    _io_info.name_to_output.emplace(name, IOIndex{_io_info.outputs.size()});
  _io_info.outputs.append(ind);
}

IOIndex Graph::getInputIndex(const std::string &name) const
{
  auto itr = _io_info.name_to_input.find(name);
  return (itr == _io_info.name_to_input.end()) ? IOIndex{} : itr->second;
}

IOIndex Graph::getOutputIndex(const std::string &name) const
{
  auto itr = _io_info.name_to_output.find(name);
  return (itr == _io_info.name_to_output.end()) ? IOIndex{} : itr->second;
}

void Graph::verify(void) const
{
  // Call graph verifications for the MODEL phase
  {
    // Except for edge consistency, the user might have been given a bad model
    // so here it throws an execption rather than assertion.
    if (!verifier::InputOutputChecker().verify(*this))
      throw std::runtime_error{"One of model input and output operands does not exist."};
    if (!verifier::DAGChecker().verify(*this))
      throw std::runtime_error{"The graph is cyclic."};
    assert(verifier::EdgeChecker().verify(*this));
  }

  // Check shape independent operation feature
  // - Operand type
  // - Shape independent parameter
  OperationValidator{*this}();
}

void Graph::initializeUseDef()
{
  operations().iterate([&](const OperationIndex &index, const IOperation &node) -> void {
    auto outputs = node.getOutputs();
    for (auto output : outputs | ir::Remove::UNDEFINED)
    {
      operands().at(output).setDef(index);
    }

    for (auto input : node.getInputs() | ir::Remove::UNDEFINED)
    {
      operands().at(input).insertUse(index);
    }
  });
}

std::vector<ir::OperationIndex> Graph::topolSortOperations() const
{
  std::vector<ir::OperationIndex> ret;
  util::Set<ir::OperationIndex> unvisited;
  operations().iterate(
    [&](const ir::OperationIndex &index, const ir::IOperation &) { unvisited.add(index); });

  std::function<void(const ir::OperationIndex &, const ir::IOperation &)> dfs =
    [&](const ir::OperationIndex &index, const ir::IOperation &op) -> void {
    if (!unvisited.contains(index))
      return;
    unvisited.remove(index);

    for (const auto output : op.getOutputs() | ir::Remove::DUPLICATED | ir::Remove::UNDEFINED)
    {
      const auto &operand = operands().at(output);
      for (const auto &use : operand.getUses())
      {
        dfs(use, operations().at(use));
      }
    }
    ret.push_back(index);
  };
  operations().iterate(dfs);

  assert(unvisited.empty()); // All of the nodes must have been visited
  // Reversing Postorder DFS result to make it sorted in topoligical order
  std::reverse(ret.begin(), ret.end());
  return ret;
}

} // namespace ir
} // namespace onert
