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
#include "PermuteTensorsLayer.h"

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
      _executor_map{executor_map}
{
  // At this point, executor_map may not have executors of cond subg and body subg
  _src_tensors = input_tensors;
  _dst_tensors = output_tensors;
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

  PermuteTensorsLayer permute_op_input_to_cond_input{_src_tensors, cond_input_tensors, _ranks};
  PermuteTensorsLayer permute_cond_input_to_body_input{cond_input_tensors, body_input_tensors,
                                                       _ranks};
  PermuteTensorsLayer permute_body_output_to_cond_input{body_output_tensors, cond_input_tensors,
                                                        _ranks};
  PermuteTensorsLayer permute_cond_input_to_op_output{cond_input_tensors, _dst_tensors, _ranks};

  // Remove copying of unused tensor
  permute_op_input_to_cond_input.prepare();
  permute_cond_input_to_body_input.prepare();
  permute_body_output_to_cond_input.prepare();
  permute_cond_input_to_op_output.prepare();

  permute_op_input_to_cond_input.run();
  cond_exec->execute();

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
    permute_cond_input_to_body_input.run();
    body_exec->execute();
    permute_body_output_to_cond_input.run();
    cond_exec->execute();
  }
  permute_cond_input_to_op_output.run();
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
