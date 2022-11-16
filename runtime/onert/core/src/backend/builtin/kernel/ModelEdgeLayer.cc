/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModelEdgeLayer.h"

#include "PermuteLayer.h"
#include "../IOTensor.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace kernel
{

ModelEdgeLayer::ModelEdgeLayer(const std::vector<backend::ITensor *> input_tensors,
                               const ir::ModelIndex &to_model_index,
                               const ir::SubgraphIndex &to_subg_index,
                               const ir::IOIndex &to_io_index, exec::Executors *executors,
                               const std::shared_ptr<ExternalContext> &external_context)
  : _input_tensors{input_tensors}, _to_model_index{to_model_index}, _to_subg_index{to_subg_index},
    _to_io_index{to_io_index}, _executors{executors}, _external_context{external_context}
{
  // At this point, executors may not have the executor of to_model_index and to_subg_index
  assert(input_tensors.size() == 1);
}

void ModelEdgeLayer::run()
{
  std::vector<ITensor *> to_tensors;
  auto to_subg_exec = _executors->at(_to_model_index, _to_subg_index);
  auto to_tensor = to_subg_exec->getInputTensors().at(_to_io_index.value());
  to_tensors.emplace_back(to_tensor);

  PermuteLayer copy_tensor_to_differnt_model{_input_tensors, to_tensors, _external_context};
  copy_tensor_to_differnt_model.run();
}

} // namespace kernel
} // namespace builtin
} // namespace backend
} // namespace onert
