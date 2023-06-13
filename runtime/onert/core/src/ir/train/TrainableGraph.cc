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

#include <algorithm>
#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace ir
{
namespace train
{

TrainableGraph::TrainableGraph() : _graph{} {}

TrainableGraph::TrainableGraph(const TrainableGraph &tgraph) : _graph{tgraph.graph()} {}

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

IOIndex TrainableGraph::getInputIndex(const std::string &name) const
{
  return _graph.getInputIndex(name);
}

IOIndex TrainableGraph::getOutputIndex(const std::string &name) const
{
  return _graph.getOutputIndex(name);
}

void TrainableGraph::addInput(const OperandIndex &ind, const std::string &name)
{
  _graph.addInput(ind, name);
}

void TrainableGraph::addOutput(const OperandIndex &ind, const std::string &name)
{
  _graph.addOutput(ind, name);
}

void TrainableGraph::verify(void) { _graph.verify(); }

void TrainableGraph::removeOperand(const OperandIndex &ind) { _graph.removeOperand(ind); }

void TrainableGraph::setLayout(Layout layout) { _graph.setLayout(layout); }

void TrainableGraph::setGraphIO(const GraphIO &io_info)
{
  assert(_graph.io_info().inputs.size() == 0);
  assert(_graph.io_info().outputs.size() == 0);
  assert(_graph.io_info().name_to_input.size() == 0);
  assert(_graph.io_info().name_to_output.size() == 0);

  auto appendIOInfo = [&](const OperandIndexSequence &sequance,
                          const std::unordered_map<std::string, IOIndex> &name_map) {
    assert(sequance.size() == name_map.size());
    for (uint32_t i = 0; i < sequance.size(); ++i)
    {
      const auto index = sequance.at(i);
      auto it = std::find_if(name_map.begin(), name_map.end(),
                             [&](const std::pair<std::string, IOIndex> &pair) {
                               const auto &io_index = pair.second;
                               return i == io_index.value();
                             });
      const auto &name = it->first;
      _graph.addInput(index, name);
    }
  };
  appendIOInfo(io_info.inputs, io_info.name_to_input);
  appendIOInfo(io_info.outputs, io_info.name_to_output);
}

const ITrainableOperation &TrainableGraph::operation(OperationIndex index) const
{
  return dynamic_cast<const ITrainableOperation &>(_graph.operations().at(index));
}

} // namespace train
} // namespace ir
} // namespace onert
