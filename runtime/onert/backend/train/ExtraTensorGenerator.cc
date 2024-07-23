/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExtraTensorGenerator.h"

#include "ExtraTensorIndex.h"

#include <ir/Operations.h>
#include <util/logging.h>
#include <memory>

namespace onert
{
namespace backend
{
namespace train
{

ExtraTensorGenerator::ExtraTensorGenerator(const ir::train::TrainableGraph *tgraph,
                                           std::shared_ptr<TensorBuilder> &tensor_builder,
                                           std::shared_ptr<ITensorRegistry> &tensor_registry)
  : _tgraph(tgraph), _tensor_builder(tensor_builder)
{
  _tensor_reg = std::dynamic_pointer_cast<TensorRegistry>(tensor_registry);
}

void ExtraTensorGenerator::register_tensors(ir::OperationIndex op_idx, ExtraTensorRequests &&reqs)
{
  // save request, _idx_to_reuqests will be used for memory planning
  if (reqs.size() == 0)
    return;

  // _idx_to_requests[op_idx] = reqs;
  _idx_to_requests.insert({op_idx, reqs});
  auto &operations = _tgraph->operations();

  for (size_t i = 0; i < reqs.size(); i++)
  {
    // register tensor
    ExtraTensorIndex tensor_idx(op_idx, i);
    _tensor_builder->registerExtraTensorInfo(tensor_idx, reqs[i].info);

    std::stringstream op_info;
    op_info << op_idx << "_" << operations.at(op_idx).name();
    VERBOSE(ExtraTensorGenerator) << "register (idx:" << tensor_idx << ") requested from "
                                  << op_info.str() << std::endl;

    // return registered tensor
    auto generated_tensor = _tensor_reg->getExtraTensor(tensor_idx);
    *reqs[i].address = generated_tensor;
  }
  return;
}

void ExtraTensorGenerator::plan()
{
  // forwarding order
  const auto f_order = _tgraph->topolSortOperations();
  for (const auto &op_index : f_order)
  {
    auto &reqs = _idx_to_requests[op_index];
    for (auto i = 0u; i < reqs.size(); ++i)
    {
      auto &lt = reqs[i].lifetime;
      if (lt == ExtraTensorLifeTime::FORWARD_TO_BACKWARD)
        _tensor_builder->notifyFirstUse(ExtraTensorIndex(op_index, i));
    }
  }

  // backwarding order
  const auto b_order = _tgraph->essentialBackwardOrder();
  for (const auto &op_index : b_order)
  {
    auto &reqs = _idx_to_requests[op_index];

    for (auto i = 0u; i < reqs.size(); ++i)
    {
      auto &lt = reqs[i].lifetime;
      if (lt == ExtraTensorLifeTime::BACKWARD)
        _tensor_builder->notifyFirstUse(ExtraTensorIndex(op_index, i));
    }

    for (auto i = 0u; i < reqs.size(); ++i)
    {
      auto &lt = reqs[i].lifetime;
      if (lt == ExtraTensorLifeTime::FORWARD_TO_BACKWARD || lt == ExtraTensorLifeTime::BACKWARD)
        _tensor_builder->notifyLastUse(ExtraTensorIndex(op_index, i));
    }
  }
}

void ExtraTensorGenerator::allocate() { _tensor_builder->allocateExtra(); }

} // namespace train
} // namespace backend
} // namespace onert
