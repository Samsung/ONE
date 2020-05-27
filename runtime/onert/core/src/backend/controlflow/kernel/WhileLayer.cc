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

#include "WhileLayer.h"

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

WhileLayer::WhileLayer(std::vector<std::shared_ptr<backend::ITensor>> input_tensors,
                       std::vector<std::shared_ptr<backend::ITensor>> output_tensors,
                       const ir::SubgraphIndex &cond_subg_index,
                       const ir::SubgraphIndex &body_subg_index,
                       const std::shared_ptr<exec::ExecutorMap> &executor_map)
    : _cond_subg_index{cond_subg_index}, _body_subg_index{body_subg_index},
      _input_tensors{input_tensors}, _output_tensors{output_tensors}, _executor_map{executor_map}
{
  // At this point, executor_map may not have executors of cond subg and body subg
  for (size_t i = 0; i < input_tensors.size(); ++i)
  {
    auto rank = input_tensors.at(i)->num_dimensions();
    // TODO Remove this when applying dynamic tensor
    assert(rank == output_tensors.at(i)->num_dimensions());
    _ranks.emplace_back(rank);
  }
}

void WhileLayer::run()
{
  // TODO Support dynamic tensor
  // Copy _src_tensors -> inputs of cond subg
  // Run cond subg
  // Start loop while output of cond subg is ture
  // // Copy cond subg inputs -> body subg inputs
  // // Run body subg
  // // Copy body subg outputs -> cond subg inputs
  // // Run cond subg
  // Copy cond subg inputs -> _dst_tensors
  auto cond_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_cond_subg_index).get());
  auto body_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_body_subg_index).get());
  if ((cond_exec == nullptr) || (body_exec == nullptr))
  {
    throw std::runtime_error{"While: Invalid condition or body"};
  }

  const auto &cond_input_tensors = cond_exec->getInputTensors();
  const auto &body_input_tensors = body_exec->getInputTensors();
  const auto &body_output_tensors = body_exec->getOutputTensors();

  const auto permute_op_input_to_cond_input =
      std::make_shared<PermuteLayer>(_input_tensors, cond_input_tensors, _ranks);
  const auto permute_cond_input_to_body_input =
      std::make_shared<PermuteLayer>(cond_input_tensors, body_input_tensors, _ranks);
  const auto permute_body_output_to_cond_input =
      std::make_shared<PermuteLayer>(body_output_tensors, cond_input_tensors, _ranks);
  const auto permute_cond_input_to_op_output =
      std::make_shared<PermuteLayer>(cond_input_tensors, _output_tensors, _ranks);

  // Remove copying of unused tensor
  permute_op_input_to_cond_input->prepare();
  permute_cond_input_to_body_input->prepare();
  permute_body_output_to_cond_input->prepare();
  permute_cond_input_to_op_output->prepare();

  cond_exec->execute(_input_tensors, permute_op_input_to_cond_input);

  assert(cond_exec->getOutputTensors().size() == 1);
  auto &cond_output_tensor = cond_exec->getOutputTensors().at(0);
  auto getResultCond = [](backend::ITensor *tensor) -> bool {
    bool ret = false;
    tensor->access([&](ITensor &tensor) { ret = *reinterpret_cast<bool *>(tensor.buffer()); });
    return ret;
  };

  // Loop while Cond subgraph's output is true
  while (getResultCond(cond_output_tensor.get()))
  {
    body_exec->execute(cond_input_tensors, permute_cond_input_to_body_input);
    cond_exec->execute(body_output_tensors, permute_body_output_to_cond_input);
  }
  permute_cond_input_to_op_output->run();
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
