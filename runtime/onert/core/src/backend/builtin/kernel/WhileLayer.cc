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

#include "PermuteLayer.h"
#include "../../../exec/ExecutorBase.h"

#include <misc/polymorphic_downcast.h>

#include <algorithm>

namespace onert::backend::builtin::kernel
{

WhileLayer::WhileLayer(const std::vector<backend::IPortableTensor *> input_tensors,
                       const std::vector<backend::IPortableTensor *> output_tensors,
                       const ir::SubgraphIndex &cond_subg_index,
                       const ir::SubgraphIndex &body_subg_index, exec::IExecutors *executors,
                       const ir::ModelIndex &model_index,
                       basic::DynamicMemoryManager *dyn_memory_manager,
                       const std::shared_ptr<ExternalContext> &external_context)
  : _cond_subg_index{cond_subg_index}, _body_subg_index{body_subg_index},
    _input_tensors{input_tensors}, _output_tensors{output_tensors}, _executors{executors},
    _model_index{model_index}, _dyn_memory_manager{dyn_memory_manager},
    _external_context{external_context}
{
  // At this point, executors may not have executors of cond subg and body subg
}

void WhileLayer::run()
{
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
  auto cond_exec = _executors->at(_model_index, _cond_subg_index);
  auto body_exec = _executors->at(_model_index, _body_subg_index);

  // Need a temp tensor to hold the cond subgraph output
  assert(cond_exec->outputSize() == 1);
  auto cond_output_tensor = [&]() {
    auto tensor = std::make_unique<Tensor>(cond_exec->outputInfo(0), _dyn_memory_manager);
    tensor->set_dynamic();
    tensor->setBuffer(_dyn_memory_manager->allocate(tensor.get(), tensor->total_size()));
    return tensor;
  }();

  VERBOSE(While) << "Call to $" << _cond_subg_index << " (cond)" << std::endl;
  const auto &options = _executors->entryExecutor()->currentOptions();
  cond_exec->execute(_input_tensors, {cond_output_tensor.get()}, options);
  VERBOSE(While) << "Return from $" << _cond_subg_index << std::endl;

  auto getResultCond = [](backend::ITensor *tensor) -> bool {
    bool ret = false;
    tensor->access([&](ITensor &tensor) { ret = *reinterpret_cast<bool *>(tensor.buffer()); });
    return ret;
  };

  std::vector<ITensor *> op_inputs(_input_tensors.begin(), _input_tensors.end());
  std::vector<ITensor *> op_outputs(_output_tensors.begin(), _output_tensors.end());
  std::vector<ir::PermuteType> permute_types;
  // Layout in graph is always NHWC, so layout is not changed
  for (uint32_t i = 0; i < op_outputs.size(); i++)
    permute_types.emplace_back(ir::PermuteType::SAME);
  // Copying body inputs to outputs when the loop body is never executed
  if (!getResultCond(cond_output_tensor.get()))
  {
    PermuteLayer copy_body_inputs_to_op_outputs{op_inputs, op_outputs, permute_types,
                                                _external_context};
    copy_body_inputs_to_op_outputs.run();
    return;
  }

  // Need some temp tensors to hold the body subgraph output
  std::vector<std::unique_ptr<Tensor>> temp_outputs_o;
  std::vector<IPortableTensor *> temp_outputs;
  for (uint32_t i = 0; i < body_exec->outputSize(); i++)
  {
    auto tensor = std::make_unique<Tensor>(body_exec->outputInfo(i), _dyn_memory_manager);
    tensor->set_dynamic();
    tensor->setBuffer(_dyn_memory_manager->allocate(tensor.get(), tensor->total_size()));
    temp_outputs.push_back(tensor.get());
    temp_outputs_o.push_back(std::move(tensor));
  }

  std::vector<ITensor *> body_outputs(temp_outputs.begin(), temp_outputs.end());
  PermuteLayer copy_body_outputs_to_op_outputs{body_outputs, op_outputs, permute_types,
                                               _external_context};

  const auto body_execute_with_op_inputs = [&]() {
    VERBOSE(While) << "Call to $" << _body_subg_index << " (body)" << std::endl;
    body_exec->execute(_input_tensors, temp_outputs, options);
    VERBOSE(While) << "Return from $" << _body_subg_index << std::endl;
  };

  const auto body_execute_with_body_outputs = [&]() {
    VERBOSE(While) << "Call to $" << _body_subg_index << " (body)" << std::endl;
    body_exec->execute(_output_tensors, temp_outputs, options);
    VERBOSE(While) << "Return from $" << _body_subg_index << std::endl;
  };

  std::function<void()> body_execute = body_execute_with_op_inputs;
  const auto cond_execute = [&]() {
    VERBOSE(While) << "Call to $" << _cond_subg_index << " (cond)" << std::endl;
    cond_exec->execute(_output_tensors, {cond_output_tensor.get()}, options);
    VERBOSE(While) << "Return from $" << _cond_subg_index << std::endl;
  };

  // Loop while Cond subgraph's output is true
  while (getResultCond(cond_output_tensor.get()))
  {
    body_execute();
    copy_body_outputs_to_op_outputs.run();
    cond_execute();
    body_execute = body_execute_with_body_outputs;
  }

  // Clean-up the temp tensors
  _dyn_memory_manager->deallocate(cond_output_tensor.get());
  for (auto &&tensor : temp_outputs)
  {
    _dyn_memory_manager->deallocate(tensor);
  }
}

} // namespace onert::backend::builtin::kernel
