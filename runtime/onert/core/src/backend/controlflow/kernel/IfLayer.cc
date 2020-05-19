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
#include "PermuteTensorsLayer.h"

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

  std::vector<size_t> input_ranks;
  for (size_t i = 0; i < _input_tensors.size(); ++i)
  {
    auto rank = _input_tensors.at(i)->num_dimensions();
    input_ranks.emplace_back(rank);
  }
  std::vector<size_t> output_ranks;
  for (size_t i = 0; i < _output_tensors.size(); ++i)
  {
    auto rank = _output_tensors.at(i)->num_dimensions();
    output_ranks.emplace_back(rank);
  }

  if (getResultCond(_cond_tensor.get()))
  {
    auto then_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_then_subg_index).get());
    if (then_exec == nullptr)
    {
      throw std::runtime_error{"If: Invalid then subgraph"};
    }

    const auto &then_input_tensors = then_exec->getInputTensors();
    const auto &then_output_tensors = then_exec->getOutputTensors();
    PermuteTensorsLayer permute_op_input_to_then_input{_input_tensors, then_input_tensors,
                                                       input_ranks};
    PermuteTensorsLayer permute_then_output_to_op_output{then_output_tensors, _output_tensors,
                                                         output_ranks};

    permute_op_input_to_then_input.run();
    then_exec->execute();
    permute_then_output_to_op_output.run();
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
    PermuteTensorsLayer permute_op_input_to_else_input{_input_tensors, else_input_tensors,
                                                       input_ranks};
    PermuteTensorsLayer permute_else_output_to_op_output{else_output_tensors, _output_tensors,
                                                         output_ranks};

    permute_op_input_to_else_input.run();
    else_exec->execute();
    permute_else_output_to_op_output.run();
  }
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
