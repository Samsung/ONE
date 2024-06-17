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

namespace onert
{
namespace backend
{
namespace builtin
{
namespace kernel
{

IfLayer::IfLayer(backend::IPortableTensor *cond_tensor,
                 const std::vector<backend::IPortableTensor *> input_tensors,
                 const std::vector<backend::IPortableTensor *> output_tensors,
                 const ir::SubgraphIndex &then_subg_index, const ir::SubgraphIndex &else_subg_index,
                 exec::IExecutors *executors, const ir::ModelIndex &model_index,
                 const std::shared_ptr<ExternalContext> &external_context)
  : _cond_tensor{cond_tensor}, _input_tensors{input_tensors}, _output_tensors{output_tensors},
    _then_subg_index{then_subg_index}, _else_subg_index{else_subg_index}, _executors{executors},
    _model_index{model_index}, _external_context{external_context}
{
  // At this point, executors may not have executors of then subg and else subg
}

void IfLayer::run()
{
  // Check condition
  // // If true
  // // // Set _input_tensors -> then-subg's inputs
  // // // Set outputs of then-subg -> _output_tensors
  // // // Run then-subg
  // // Else
  // // // Set _input_tensors -> else-subg's inputs
  // // // Set outputs of else-subg -> _output_tensors
  // // // Run else-subg

  auto getResultCond = [](backend::IPortableTensor *tensor) -> bool {
    bool ret = false;
    tensor->access([&](ITensor &tensor) { ret = *reinterpret_cast<bool *>(tensor.buffer()); });
    return ret;
  };

  exec::IExecutor *subg_exec = nullptr;
  bool cond_result = getResultCond(_cond_tensor);
  if (cond_result)
  {
    VERBOSE(If) << "Call to $" << _then_subg_index << " (then)" << std::endl;
    subg_exec = _executors->at(_model_index, _then_subg_index);
  }
  else
  {
    VERBOSE(If) << "Call to $" << _else_subg_index << " (else)" << std::endl;
    subg_exec = _executors->at(_model_index, _else_subg_index);
  }

  subg_exec->execute(_input_tensors, _output_tensors,
                     _executors->entryExecutor()->currentOptions());
  VERBOSE(If) << "Return from $" << (cond_result ? _then_subg_index : _else_subg_index)
              << std::endl;
}

} // namespace kernel
} // namespace builtin
} // namespace backend
} // namespace onert
