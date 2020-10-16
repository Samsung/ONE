/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "IfLayer.h"

#include <backend/ITensor.h>
#include "exec/ExecutorBase.h"
#include <misc/polymorphic_downcast.h>
#include "PermuteLayer.h"

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

IfLayer::IfLayer(backend::IPortableTensor *cond_tensor,
                 const std::vector<backend::IPortableTensor *> input_tensors,
                 const std::vector<backend::IPortableTensor *> output_tensors,
                 const ir::OperandIndexSequence &output_indices, const ir::Graph &graph,
                 const ir::SubgraphIndex &then_subg_index, const ir::SubgraphIndex &else_subg_index,
                 exec::ExecutorMap *executor_map)
    : _cond_tensor{cond_tensor}, _input_tensors{input_tensors}, _output_tensors{output_tensors},
      _output_indices{output_indices}, _graph{graph}, _then_subg_index{then_subg_index},
      _else_subg_index{else_subg_index}, _executor_map{executor_map}
{
  // At this point, executor_map may not have executors of then subg and else subg
}

void IfLayer::run()
{
  // Check condition
  // // If true
  // // // Copy _input_tensors -> then subg's inputs
  // // // Run then subg
  // // // Copy outputs of then subg -> _output_tensors
  // // Else
  // // // Copy _input_tensors -> else subg's inputs if false
  // // // Run else subg
  // // // Copy outputs of else subg -> _output_tensors
  auto getResultCond = [](backend::IPortableTensor *tensor) -> bool {
    bool ret = false;
    tensor->access([&](ITensor &tensor) { ret = *reinterpret_cast<bool *>(tensor.buffer()); });
    return ret;
  };

  exec::ExecutorBase *subg_exec = nullptr;
  bool cond_result = getResultCond(_cond_tensor);
  if (cond_result)
  {
    VERBOSE(If) << "Call to $" << _then_subg_index << " (then)" << std::endl;
    subg_exec = nnfw::misc::polymorphic_downcast<exec::ExecutorBase *>(
        _executor_map->at(_then_subg_index).get());
  }
  else
  {
    VERBOSE(If) << "Call to $" << _else_subg_index << " (else)" << std::endl;
    subg_exec = nnfw::misc::polymorphic_downcast<exec::ExecutorBase *>(
        _executor_map->at(_else_subg_index).get());
  }

  const auto &subg_graph = subg_exec->graph();

  std::vector<backend::ITensor *> src_tensors;
  std::vector<backend::ITensor *> dst_tensors;
  // Add tensors used in subgraph or contained in outputs of subgraph
  assert(subg_graph.getInputs().size() == _input_tensors.size());
  assert(subg_graph.getInputs().size() == subg_exec->getInputTensors().size());
  for (uint32_t i = 0; i < subg_graph.getInputs().size(); ++i)
  {
    const auto &subg_input_index = subg_graph.getInputs().at(i);
    const auto &subg_input = subg_graph.operands().at(subg_input_index);
    if (subg_input.getUses().size() > 0 || subg_graph.getOutputs().contains(subg_input_index))
    {
      src_tensors.emplace_back(_input_tensors.at(i));
      dst_tensors.emplace_back(subg_exec->getInputTensors().at(i));
    }
  }
  const auto permute_op_input_to_subg_input =
      std::make_shared<PermuteLayer>(src_tensors, dst_tensors);

  // Add tensors used as output of operation or contained in outputs of operation
  src_tensors.clear();
  dst_tensors.clear();
  assert(_output_indices.size() == subg_exec->getOutputTensors().size());
  assert(_output_indices.size() == _output_tensors.size());
  for (uint32_t i = 0; i < _output_indices.size(); ++i)
  {
    const auto &output_index = _output_indices.at(i);
    const auto &output = _graph.operands().at(output_index);
    if (output.getUses().size() > 0 || _graph.getOutputs().contains(output_index))
    {
      src_tensors.emplace_back(subg_exec->getOutputTensors().at(i));
      dst_tensors.emplace_back(_output_tensors.at(i));
    }
  }
  const auto permute_subg_output_to_op_output =
      std::make_shared<PermuteLayer>(src_tensors, dst_tensors);

  // Remove copying of unused tensor
  permute_subg_output_to_op_output->prepare();

  // Copy & run
  subg_exec->execute(_input_tensors, _output_tensors);
  // permute_subg_output_to_op_output->run();
  VERBOSE(If) << "Return from $" << (cond_result ? _then_subg_index : _else_subg_index)
              << std::endl;
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
