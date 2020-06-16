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
#include "PermuteLayer.h"

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

IfLayer::IfLayer(const std::shared_ptr<backend::ITensor> &cond_tensor,
                 std::vector<std::shared_ptr<backend::ITensor>> input_tensors,
                 std::vector<std::shared_ptr<backend::ITensor>> output_tensors,
                 const exec::DynAllocInfoMap &outputs_dyn_alloc_info,
                 const ir::SubgraphIndex &then_subg_index, const ir::SubgraphIndex &else_subg_index,
                 const std::shared_ptr<exec::ExecutorMap> &executor_map)
    : _cond_tensor{cond_tensor}, _input_tensors{input_tensors}, _output_tensors{output_tensors},
      _outputs_dyn_alloc_info{outputs_dyn_alloc_info}, _then_subg_index{then_subg_index},
      _else_subg_index{else_subg_index}, _executor_map{executor_map}
{
  // At this point, executor_map may not have executors of then subg and else subg
}

void IfLayer::run()
{
  // Check condition
  // // If true
  // // // Copy _src_tensors -> then subg's inputs
  // // // Run then subg
  // // // Copy outputs of then subg -> _dst_tensors
  // // Else
  // // // Copy _src_tensors -> else subg's inputs if false
  // // // Run else subg
  // // // Copy outputs of else subg -> _dst_tensors
  auto getResultCond = [](backend::ITensor *tensor) -> bool {
    bool ret = false;
    tensor->access([&](ITensor &tensor) { ret = *reinterpret_cast<bool *>(tensor.buffer()); });
    return ret;
  };

  exec::ExecutorBase *subg_exec = nullptr;
  if (getResultCond(_cond_tensor.get()))
  {
    subg_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_then_subg_index).get());
    if (subg_exec == nullptr)
    {
      throw std::runtime_error{"If: Invalid then subgraph"};
    }
  }
  else
  {
    subg_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_else_subg_index).get());
    if (subg_exec == nullptr)
    {
      throw std::runtime_error{"If: Invalid else subgraph"};
    }
  }

  const auto &subg_input_tensors = subg_exec->getInputTensors();
  const auto &subg_inputs_dyn_alloc_info = subg_exec->getInputsDynamicAllocInfo();

  const auto permute_op_input_to_subg_input = std::make_shared<PermuteLayer>(
      _input_tensors, subg_input_tensors, subg_inputs_dyn_alloc_info);

  const auto &subg_output_tensors = subg_exec->getOutputTensors();
  const auto permute_subg_output_to_op_output =
      std::make_shared<PermuteLayer>(subg_output_tensors, _output_tensors, _outputs_dyn_alloc_info);

  // Remove copying of unused tensor
  permute_op_input_to_subg_input->prepare();
  permute_subg_output_to_op_output->prepare();

  // Copy & run
  assert(_input_tensors.size() == subg_input_tensors.size());
  subg_exec->execute(_input_tensors, permute_op_input_to_subg_input);
  assert(_output_tensors.size() == subg_output_tensors.size());
  for (size_t i = 0; i < _output_tensors.size(); ++i)
  {
    const auto output_tensor = _output_tensors.at(i);
    const auto orig_output_shape = output_tensor->getShape();
    const auto changed_output_shape = subg_output_tensors.at(i)->getShape();
    if (orig_output_shape != changed_output_shape &&
        _outputs_dyn_alloc_info.find(output_tensor) != _outputs_dyn_alloc_info.end())
    {
      output_tensor->set_dynamic();
    }
  }
  permute_subg_output_to_op_output->run();
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
