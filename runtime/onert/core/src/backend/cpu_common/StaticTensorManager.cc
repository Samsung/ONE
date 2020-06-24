/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "backend/cpu_common/StaticTensorManager.h"

#include <util/logging.h>

namespace onert
{
namespace backend
{
namespace cpu_common
{

StaticTensorManager::StaticTensorManager(const std::shared_ptr<TensorRegistry> &reg)
    : _const_mgr{new DynamicMemoryManager()}, _nonconst_mgr{new MemoryManager()}, _tensors{reg}
{
  // DO NOTHING
}

void StaticTensorManager::allocateConsts(void)
{
  for (auto &pair : _tensors->managed_tensors())
  {
    const auto &ind = pair.first;
    auto tensor = pair.second;
    if (_as_constants[ind])
    {
      auto mem_alloc = _const_mgr->allocate(ind, tensor->total_size());
      tensor->setBuffer(mem_alloc);
      auto buffer = mem_alloc->base();
      VERBOSE(CPU_StaticTensorManager) << "CONSTANT TENSOR(#" << ind.value()
                                       << "): " << static_cast<void *>(buffer)
                                       << "size : " << tensor->total_size() << std::endl;
    }
  }
}

void StaticTensorManager::allocateNonconsts(void)
{
  _nonconst_mgr->allocate();

  for (auto &pair : _tensors->managed_tensors())
  {
    const auto &ind = pair.first;
    auto tensor = pair.second;
    if (!_as_constants[ind] && !tensor->is_dynamic())
    {
      auto *buffer = _nonconst_mgr->getBuffer(ind);
      tensor->setBuffer(buffer);

      VERBOSE(CPU_StaticTensorManager) << "TENSOR(#" << ind.value()
                                       << "): " << static_cast<void *>(buffer) << std::endl;
    }
  }
}

void StaticTensorManager::deallocateConsts(void) { _const_mgr->deallocate(); }

void StaticTensorManager::deallocateNonconsts(void) { _nonconst_mgr->deallocate(); }

void StaticTensorManager::buildTensor(const ir::OperandIndex &ind,
                                      const ir::OperandInfo &tensor_info, ir::Layout backend_layout,
                                      bool as_const)
{
  assert(!_tensors->getManagedTensor(ind));
  auto tensor = std::make_shared<Tensor>(tensor_info, backend_layout);
  _tensors->setManagedTensor(ind, tensor);
  _as_constants[ind] = as_const;
}

void StaticTensorManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  assert(_tensors->getManagedTensor(ind));

  // This method is called only when a tensor has proper shape
  assert(!_tensors->getManagedTensor(ind)->is_dynamic());

  if (!_as_constants[ind])
    _nonconst_mgr->claimPlan(ind, size);
}

void StaticTensorManager::releasePlan(const ir::OperandIndex &ind)
{
  assert(_tensors->getManagedTensor(ind));

  // This method is called only when a tensor has proper shape
  assert(!_tensors->getManagedTensor(ind)->is_dynamic());

  if (!_as_constants[ind])
    _nonconst_mgr->releasePlan(ind);
}

void StaticTensorManager::iterate(const std::function<void(const ir::OperandIndex &)> &fn)
{
  for (const auto &it : _tensors->managed_tensors())
    fn(it.first);
}

} // namespace cpu_common
} // namespace backend
} // namespace onert
