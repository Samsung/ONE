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

#include <util/logging.h>

#include <cassert>

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{

TensorBuilder::TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg)
  : _tensor_reg{tensor_reg}, _tensor_mgr{new TensorManager(_tensor_reg)}
{
  /* empty */
}

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &index, const ir::OperandInfo &info,
                                       ir::Layout layout)
{
  _tensor_info_map.emplace(index, info);

  // Train backend supports only one layout as NHWC
  assert(layout == ir::Layout::NHWC);
  assert(!info.isDynamic());

  auto tensor = std::make_unique<Tensor>(info, layout);
  _tensor_reg->base_reg()->setNonConstTensor(index, std::move(tensor));
}

void TensorBuilder::notifyFirstUse(const ir::OperandIndex &index)
{
  // TODO Enhance the way of checking user tensors
  if (_tensor_info_map.find(index) == _tensor_info_map.end()) // Do not proceed for user tensors
    return;

  _tensor_mgr->claimNonConstPlan(index);
}

void TensorBuilder::notifyLastUse(const ir::OperandIndex &index)
{
  // TODO Enhance the way of checking user tensors
  if (_tensor_info_map.find(index) == _tensor_info_map.end()) // Do not proceed for user tensors
    return;

  _tensor_mgr->releaseNonConstPlan(index);
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &index) const
{
  // User tensors are not registered in _tensor_info_map but objects for them are exist
  // in the tensor registry.
  // TODO Enhance the way of checking user tensors
  if (_tensor_reg->getITensor(index))
    return true;
  return _tensor_info_map.find(index) != _tensor_info_map.end();
}

void TensorBuilder::allocate(void) { _tensor_mgr->allocateNonConstTensors(); }

} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
