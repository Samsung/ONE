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
                 const ir::SubgraphIndex &then_subg_index, const ir::SubgraphIndex &else_subg_index,
                 const std::shared_ptr<exec::ExecutorMap> &executor_map)
    : _cond_tensor{cond_tensor}, _input_tensors{input_tensors}, _output_tensors{output_tensors},
      _then_subg_index{then_subg_index}, _else_subg_index{else_subg_index},
      _executor_map{executor_map}
{
  // At this point, executor_map may not have executors of then subg and else subg
}

void IfLayer::run()
{
  // TODO Support dynamic tensor
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

  if (getResultCond(_cond_tensor.get()))
  {
    auto then_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_then_subg_index).get());
    if (then_exec == nullptr)
    {
      throw std::runtime_error{"If: Invalid then subgraph"};
    }

    const auto &then_input_tensors = then_exec->getInputTensors();
    const auto &then_output_tensors = then_exec->getOutputTensors();
    const auto permute_op_input_to_then_input =
        std::make_shared<PermuteLayer>(_input_tensors, then_input_tensors);
    const auto permute_then_output_to_op_output =
        std::make_shared<PermuteLayer>(then_output_tensors, _output_tensors);

    // Remove copying of unused tensor
    permute_op_input_to_then_input->prepare();
    permute_then_output_to_op_output->prepare();

    // Copy & run
    then_exec->execute(_input_tensors, permute_op_input_to_then_input);
    permute_then_output_to_op_output->run();
  }
  else
  {
    auto else_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_else_subg_index).get());
    if (else_exec == nullptr)
    {
      throw std::runtime_error{"If: Invalid else subgraph"};
    }

    const auto &else_input_tensors = else_exec->getInputTensors();
    const auto &else_output_tensors = else_exec->getOutputTensors();
    const auto permute_op_input_to_else_input =
        std::make_shared<PermuteLayer>(_input_tensors, else_input_tensors);
    const auto permute_else_output_to_op_output =
        std::make_shared<PermuteLayer>(else_output_tensors, _output_tensors);

    // Remove copying of unused tensor
    permute_op_input_to_else_input->prepare();
    permute_else_output_to_op_output->prepare();

    // Copy & run
    else_exec->execute(_input_tensors, permute_op_input_to_else_input);
    permute_else_output_to_op_output->run();
  }
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
