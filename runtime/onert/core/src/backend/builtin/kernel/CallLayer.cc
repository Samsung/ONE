/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CallLayer.h"

namespace onert::backend::builtin::kernel
{

CallLayer::CallLayer(const std::vector<backend::IPortableTensor *> input_tensors,
                     const std::vector<backend::IPortableTensor *> output_tensors,
                     const ir::SubgraphIndex &callee_subg_index, exec::IExecutors *executors,
                     const ir::ModelIndex &model_index,
                     const std::shared_ptr<ExternalContext> &external_context)
  : _input_tensors{input_tensors}, _output_tensors{output_tensors},
    _callee_subg_index{callee_subg_index}, _executors{executors}, _model_index{model_index},
    _external_context{external_context}
{
  // DO NOTHING
}

void CallLayer::run()
{
  exec::IExecutor *subg_exec = _executors->at(_model_index, _callee_subg_index);
  subg_exec->execute(_input_tensors, _output_tensors,
                     _executors->entryExecutor()->currentOptions());
}

} // namespace onert::backend::builtin::kernel
