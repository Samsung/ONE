/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
namespace cpu
{

TensorBuilder::TensorBuilder()
    : _tensor_reg{new TensorRegistry()}, _static_tensor_mgr{new StaticTensorManager(_tensor_reg)},
      _dynamic_tensor_mgr{new DynamicTensorManager(_tensor_reg)}
{
  /* empty */
}

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                                       ir::Layout, bool as_const)
{
  _tensor_info_map.emplace(ind, info);

  if (as_const)
    _constants.append(ind);

  // TODO consider dynamic tensor
  _static_tensor_mgr->buildTensor(ind, info, _constants.contains(ind));
}

void TensorBuilder::notifyFirstUse(const ir::OperandIndex &ind)
{
  assert(_tensor_info_map.find(ind) != _tensor_info_map.end());
  const auto tensor_info = _tensor_info_map.at(ind);
  const auto size = tensor_info.total_size();

  // TODO consider dynamic tensor
  _static_tensor_mgr->claimPlan(ind, size);
}

void TensorBuilder::notifyLastUse(const ir::OperandIndex &ind)
{
  _static_tensor_mgr->releasePlan(ind);
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &ind) const
{
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

std::shared_ptr<ITensor> TensorBuilder::tensorAt(const ir::OperandIndex &ind)
{
  auto found = _tensor_reg->find(ind);
  if (found == _tensor_reg->end())
    return nullptr;

  return found->second;
}

void TensorBuilder::iterate(const IterateFunction &fn) { _static_tensor_mgr->iterate(fn); }

std::shared_ptr<operand::Tensor> TensorBuilder::at(const ir::OperandIndex &ind)
{
  auto found = _tensor_reg->find(ind);
  assert(found != _tensor_reg->end());
  return found->second;
}

std::unique_ptr<ITensorManager> TensorBuilder::releaseStaticTensorManager(void)
{
  return std::move(_static_tensor_mgr);
}

std::unique_ptr<ITensorManager> TensorBuilder::releaseDynamicTensorManager(void)
{
  return std::move(_dynamic_tensor_mgr);
}

} // namespace cpu
} // namespace backend
} // namespace onert
