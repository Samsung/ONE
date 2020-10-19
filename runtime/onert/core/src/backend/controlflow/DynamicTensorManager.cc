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

#include "DynamicTensorManager.h"

#include "util/logging.h"
#include "misc/polymorphic_downcast.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

DynamicTensorManager::DynamicTensorManager(const std::shared_ptr<TensorRegistry> &tensors)
    : _dynamic_mem_mgr{new cpu_common::DynamicMemoryManager()}, _tensors{tensors}
{
  // DO NOTHING
}

void DynamicTensorManager::buildTensor(const ir::OperandIndex &ind,
                                       const ir::OperandInfo &tensor_info,
                                       ir::Layout backend_layout)
{
  auto tensor =
      std::make_unique<cpu_common::Tensor>(tensor_info, backend_layout, _dynamic_mem_mgr.get());
  _tensors->setNativeOwnTensor(ind, std::move(tensor));
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
    cpu_tensor->resetBuffer();

    VERBOSE(DynamicTensorManager) << "Deallocating a tensor " << (void *)tensor
                                  << " (input of op_ind: " << op_ind.value() << ")" << std::endl;
  }
}

const ITensor *DynamicTensorManager::getRawITensor(ir::OperandIndex ind)
{
  auto ptr = _tensors->getITensor(ind);
  assert(ptr);
  return ptr;
}

} // namespace controlflow
} // namespace backend
} // namespace onert
