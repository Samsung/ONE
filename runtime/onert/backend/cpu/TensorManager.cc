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

#include "TensorManager.h"

#include <util/logging.h>

namespace onert
{
namespace backend
{
namespace cpu
{

TensorManager::TensorManager()
    : _const_mgr{new cpu_common::DynamicMemoryManager()},
      _nonconst_mgr{new cpu_common::MemoryManager()},
      _dynamic_tensor_mgr{new cpu_common::DynamicMemoryManager()}
{
  // DO NOTHING
}

void TensorManager::allocateConsts(void)
{
  for (auto &pair : _tensors)
  {
    const auto &ind = pair.first;
    auto tensor = pair.second;
    if (_as_constants[ind])
    {
      auto mem_alloc = _const_mgr->allocate(ind, tensor->total_size());
      tensor->setBuffer(mem_alloc);
      auto buffer = mem_alloc->base();
      VERBOSE(CPU_TENSORMANAGER) << "CONSTANT TENSOR(#" << ind.value()
                                 << "): " << static_cast<void *>(buffer)
                                 << "size : " << tensor->total_size() << std::endl;
    }
  }
}

void TensorManager::allocateNonconsts(void)
{
  _nonconst_mgr->allocate();

  for (auto &pair : _tensors)
  {
    const auto &ind = pair.first;
    auto tensor = pair.second;
    if (!_as_constants[ind] && !tensor->is_dynamic())
    {
      auto *buffer = _nonconst_mgr->getBuffer(ind);
      tensor->setBuffer(buffer);

      VERBOSE(CPU_TENSORMANAGER) << "TENSOR(#" << ind.value()
                                 << "): " << static_cast<void *>(buffer) << std::endl;
    }
  }
}

void TensorManager::deallocateConsts(void) { _const_mgr->deallocate(); }

void TensorManager::deallocateNonconsts(void) { _nonconst_mgr->deallocate(); }

void TensorManager::allocateDynamicTensor(const ir::OperandIndex &ind, const ir::Shape &new_shape)
{
  auto tensor = _tensors[ind];
  assert(tensor);

  tensor->info().shape(new_shape);

  auto capacity = tensor->total_size();
  auto alloc = _dynamic_tensor_mgr->allocate(ind, capacity);

  tensor->setBuffer(alloc);
}

void TensorManager::buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &tensor_info,
                                bool as_const)
{
  assert(_tensors.find(ind) == _tensors.end());
  auto tensor = std::make_shared<operand::Tensor>(tensor_info);
  _tensors[ind] = tensor;
  _as_constants[ind] = as_const;
}

void TensorManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  assert(_tensors.find(ind) != _tensors.end());

  // This method is called only when a tensor is not dynamic
  assert(!_tensors[ind]->is_dynamic());

  if (!_as_constants[ind])
    _nonconst_mgr->claimPlan(ind, size);
}

void TensorManager::releasePlan(const ir::OperandIndex &ind)
{
  assert(_tensors.find(ind) != _tensors.end());

  // This method is called only when a tensor is not dynamic
  assert(!_tensors[ind]->is_dynamic());

  if (!_as_constants[ind])
    _nonconst_mgr->releasePlan(ind);
}

std::shared_ptr<operand::Tensor> TensorManager::at(const ir::OperandIndex &ind)
{
  if (_tensors.find(ind) == _tensors.end())
    return nullptr;
  return _tensors.at(ind);
}

void TensorManager::iterate(const std::function<void(const ir::OperandIndex &)> &fn)
{
  for (const auto &it : _tensors)
    fn(it.first);
}

} // namespace cpu
} // namespace backend
} // namespace onert
