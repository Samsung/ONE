/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TensorBuilder.h"

namespace onert
{
namespace backend
{
namespace train
{

TensorBuilder::TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg,
                             const std::string planner_id)
  : _tensor_reg{tensor_reg}, _tensor_mgr{new TensorManager(_tensor_reg, planner_id)}
{
  /* empty */
}

void TensorBuilder::registerForwardTensorInfo(const ir::OperandIndex &index,
                                              const ir::OperandInfo &info, ir::Layout layout)
{
  _forward_tensor_info_map.emplace(index, info);

  // Train backend supports only one layout as NHWC
  assert(layout == ir::Layout::NHWC);
  assert(!info.isDynamic());
  _tensor_mgr->buildForwardTensor(index, info, layout, info.isConstant());
}

void TensorBuilder::registerBackwardTensorInfo(const ir::OperandIndex &index,
                                               const ir::OperandInfo &info, ir::Layout layout)
{
  _backward_tensor_info_map.emplace(index, info);

  // Train backend supports only one layout as NHWC
  assert(layout == ir::Layout::NHWC);
  assert(!info.isDynamic());
  _tensor_mgr->buildBackwardTensor(index, info, layout, info.isConstant());
}

void TensorBuilder::notifyForwardFirstUse(const ir::OperandIndex &index)
{
  // TODO Support memory plan
  _tensor_mgr->claimForwardPlan(index);
}

void TensorBuilder::notifyBackwardFirstUse(const ir::OperandIndex &index)
{
  // TODO Support memory plan
  _tensor_mgr->claimBackwardPlan(index);
}

void TensorBuilder::notifyLastUse(const ir::OperandIndex &)
{
  // TODO Support momory plan
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &index) const
{
  return _forward_tensor_info_map.find(index) != _forward_tensor_info_map.end();
}

void TensorBuilder::allocateForwardTensors(void) { _tensor_mgr->allocateForwardTensors(); }

void TensorBuilder::allocateBackwardTensors(void) { _tensor_mgr->allocateBackwardTensors(); }

} // namespace train
} // namespace backend
} // namespace onert
