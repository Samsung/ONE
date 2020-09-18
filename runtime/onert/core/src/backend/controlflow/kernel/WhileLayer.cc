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

#include <algorithm>
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

WhileLayer::WhileLayer(const std::vector<backend::ITensor *> input_tensors,
                       const std::vector<backend::ITensor *> output_tensors,
                       const ir::SubgraphIndex &cond_subg_index,
                       const ir::SubgraphIndex &body_subg_index, exec::ExecutorMap *executor_map,
                       cpu_common::DynamicMemoryManager *dyn_memory_manager)
    : _cond_subg_index{cond_subg_index}, _body_subg_index{body_subg_index},
      _input_tensors{input_tensors}, _output_tensors{output_tensors}, _executor_map{executor_map},
      _dyn_memory_manager{dyn_memory_manager}
{
  // At this point, executor_map may not have executors of cond subg and body subg
}

void WhileLayer::run()
{
  VERBOSE(WHILE) << "WHILE LAYER ===========================" << std::endl;
  // Copy "_input_tensors" -> "cond subg inputs"
  // Run cond subg
  // Start loop while output of cond subg is ture
  // // Copy "_input_tensors" -> "body subg inputs" in the first iteration, then copy "body subg
  // outputs" -> "body subg inputs" in the second or more iterations
  // // Run body subg
  // // Copy "body subg outputs" -> "cond subg inputs"
  // // Run cond subg
  // If there is no loop copy "_input_tensors" -> "_dst_tensors", else copy "cond subg inputs" ->
  // "_dst_tensors"
  auto cond_exec = nnfw::misc::polymorphic_downcast<exec::ExecutorBase *>(
      _executor_map->at(_cond_subg_index).get());
  auto body_exec = nnfw::misc::polymorphic_downcast<exec::ExecutorBase *>(
      _executor_map->at(_body_subg_index).get());

  assert(cond_exec->getOutputTensors().size() == 1);
  auto &cond_output_tensor = cond_exec->getOutputTensors().at(0);
  auto cond_output_tensor_size = std::max(cond_output_tensor->total_size(), static_cast<size_t>(1));
  auto cond_buf = std::unique_ptr<uint8_t[]>{new uint8_t(cond_output_tensor_size)};

  cond_output_tensor->setUserTensor(cond_buf.get(), cond_output_tensor_size);
  VERBOSE(While) << "Call to $" << _cond_subg_index << " (cond)" << std::endl;
  cond_exec->execute(_input_tensors, {}); // XXX output tensor is processed above
  VERBOSE(While) << "Return from $" << _cond_subg_index << std::endl;

  auto getResultCond = [](backend::ITensor *tensor) -> bool {
    bool ret = false;
    tensor->access([&](ITensor &tensor) { ret = *reinterpret_cast<bool *>(tensor.buffer()); });
    return ret;
  };

  // Copying body inputs to outputs when the loop body is never executed
  if (!getResultCond(cond_output_tensor))
  {
    PermuteLayer copy_body_inputs_to_op_outputs{_input_tensors, _output_tensors};
    copy_body_inputs_to_op_outputs.run();
    // return;
  }

  // Need some temp tensors to hold the body subgraph output
  std::vector<std::unique_ptr<Tensor>> temp_outputs_o;
  std::vector<ITensor *> temp_outputs;
  for (auto io_tensor : body_exec->getOutputTensors())
  {
    auto tensor = std::make_unique<Tensor>(io_tensor->orig_info(), io_tensor->orig_layout(),
                                           _dyn_memory_manager);
    tensor->set_dynamic();
    tensor->setBuffer(_dyn_memory_manager->allocate(tensor.get(), tensor->total_size()));
    temp_outputs.push_back(tensor.get());
    temp_outputs_o.push_back(std::move(tensor));
  }

  std::vector<ITensor *> body_outputs(body_exec->getOutputTensors().begin(),
                                      body_exec->getOutputTensors().end());
  PermuteLayer copy_body_outputs_to_op_outputs{body_outputs, _output_tensors};

  const auto body_execute_with_op_inputs = [&]() {
    VERBOSE(While) << "Call to $" << _body_subg_index << " (body)" << std::endl;
    body_exec->execute(_input_tensors, temp_outputs);
    VERBOSE(While) << "Return from $" << _body_subg_index << std::endl;
  };

  const auto body_execute_with_body_outputs = [&]() {
    VERBOSE(While) << "Call to $" << _body_subg_index << " (body)" << std::endl;
    body_exec->execute(_output_tensors, temp_outputs);
    VERBOSE(While) << "Return from $" << _body_subg_index << std::endl;
  };

  std::function<void()> body_execute = body_execute_with_op_inputs;
  const auto cond_execute = [&]() {
    cond_output_tensor->setUserTensor(cond_buf.get(), cond_output_tensor_size);
    VERBOSE(While) << "Call to $" << _cond_subg_index << " (cond)" << std::endl;
    cond_exec->execute(_output_tensors, {});
    VERBOSE(While) << "Return from $" << _cond_subg_index << std::endl;
  };

  // Loop while Cond subgraph's output is true
  while (getResultCond(cond_output_tensor))
  {
    body_execute();
    copy_body_outputs_to_op_outputs.run();
    cond_execute();
    body_execute = body_execute_with_body_outputs;
  }

  // Clean-up the temp tensors
  for (auto tensor : temp_outputs)
  {
    _dyn_memory_manager->deallocate(tensor);
  }
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
