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

#include "TensorBuilder.h"

#include <util/logging.h>

#include <cassert>

namespace onert
{
namespace backend
{
namespace controlflow
{

TensorBuilder::TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg)
    : _tensor_reg{tensor_reg},
      _dynamic_tensor_mgr{new DynamicTensorManager(_tensor_reg->base_reg())},
      _static_tensor_mgr{new cpu_common::StaticTensorManager(
          _tensor_reg->base_reg(), _dynamic_tensor_mgr->dynamic_mem_mgr().get())}
{
  /* empty */
}

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                                       ir::Layout backend_layout)
{
  _tensor_info_map.emplace(ind, info);

  _tensor_layout_map.insert({ind, backend_layout});

  if (info.isDynamic())
  {
    _dynamic_tensor_mgr->buildTensor(ind, info, _tensor_layout_map[ind]);
  }
  else
  {
    _static_tensor_mgr->buildTensor(ind, info, _tensor_layout_map[ind], info.isConstant());
  }
}

void TensorBuilder::notifyFirstUse(const ir::OperandIndex &ind)
{
  // TODO Enhance the way of checking user tensors
  if (_tensor_info_map.find(ind) == _tensor_info_map.end()) // Do not proceed for user tensors
    return;

  const auto tensor_info = _tensor_info_map.at(ind);

  if (!nativeOwnTensorAt(ind)->is_dynamic())
  {
    const auto size = tensor_info.total_size();
    _static_tensor_mgr->claimPlan(ind, size);
  }
}

void TensorBuilder::notifyLastUse(const ir::OperandIndex &ind)
{
  // TODO Enhance the way of checking user tensors
  if (_tensor_info_map.find(ind) == _tensor_info_map.end()) // Do not proceed for user tensors
    return;

  if (!nativeOwnTensorAt(ind)->is_dynamic())
  {
    _static_tensor_mgr->releasePlan(ind);
  }
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &ind) const
{
  // User tensors are not registered in _tensor_info_map but objects for them are exist
  // in the tensor registry.
  // TODO Enhance the way of checking user tensors
  if (_tensor_reg->getITensor(ind))
    return true;
  return _tensor_info_map.find(ind) != _tensor_info_map.end();
}

void TensorBuilder::prepare(void)
{
  _static_tensor_mgr->allocateConsts();
  _static_tensor_mgr->allocateNonconsts();
}

void TensorBuilder::allocate()
{
  // NOTE For now nothing to do. Allocation is done in prepare stage, which is not appropriate
  //      This is because CPU kernels require `ITensor`s to be allocated before Kernel Generation.
}

DynamicTensorManager *TensorBuilder::dynamicTensorManager(void)
{
  return _dynamic_tensor_mgr.get();
}

cpu_common::Tensor *TensorBuilder::nativeOwnTensorAt(const ir::OperandIndex &ind)
{
  return _tensor_reg->getNativeOwnTensor(ind);
}

} // namespace controlflow
} // namespace backend
} // namespace onert
