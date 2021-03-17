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

#include "backend/cpu_common/DynamicTensorManager.h"

#include "util/logging.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

DynamicTensorManager::DynamicTensorManager(const std::shared_ptr<TensorRegistry> &reg)
  : _dynamic_mem_mgr{new DynamicMemoryManager()}, _tensors{reg}
{
  // DO NOTHING
}

void DynamicTensorManager::buildTensor(const ir::OperandIndex &ind,
                                       const ir::OperandInfo &tensor_info,
                                       ir::Layout backend_layout)
{
  assert(_tensors->getNativeTensor(ind) == nullptr);
  auto tensor = std::make_unique<Tensor>(tensor_info, backend_layout, _dynamic_mem_mgr.get());
  _tensors->setNativeTensor(ind, std::move(tensor));
}

void DynamicTensorManager::planDealloc(ir::OperationIndex op_ind, backend::ITensor *tensor)
{
  _dealloc_tensor_map[op_ind].emplace(tensor);
}

void DynamicTensorManager::deallocInput(ir::OperationIndex op_ind)
{
  auto find = _dealloc_tensor_map.find(op_ind);
  if (find == _dealloc_tensor_map.end())
    return;

  auto &input_set = find->second;
  for (auto *tensor : input_set)
  {
    if (!tensor->is_dynamic())
      continue;

    _dynamic_mem_mgr->deallocate(tensor);

    auto *cpu_tensor = nnfw::misc::polymorphic_downcast<cpu_common::Tensor *>(tensor);
    cpu_tensor->deallocBuffer();

    VERBOSE(DynamicTensorManager) << "Deallocating tensor " << (void *)cpu_tensor
                                  << " (input of op_ind: " << op_ind << ")" << std::endl;
  }
}

const ITensor *DynamicTensorManager::getRawITensor(ir::OperandIndex ind)
{
  auto ptr = _tensors->getITensor(ind);
  assert(ptr);
  return ptr;
}

} // namespace cpu_common
} // namespace backend
} // namespace onert
