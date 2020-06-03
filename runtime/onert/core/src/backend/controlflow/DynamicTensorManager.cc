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

// This file will be removed or unified with backend/cpu/DynamicTensorManager.cc

#include "DynamicTensorManager.h"

#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

DynamicTensorManager::DynamicTensorManager(const std::shared_ptr<TensorRegistry> &reg)
    : _dynamic_mem_mgr{new cpu_common::DynamicMemoryManager()}, _tensors{reg}
{
  // DO NOTHING
}

void DynamicTensorManager::allocate(const ir::OperandIndex &ind, const ir::Shape &new_shape)
{
  auto tensor = (*_tensors)[ind];
  assert(tensor);

  auto allocTensorMem = [&]() {
    setShape(tensor.get(), new_shape);

    auto capacity = tensor->total_size();
    auto alloc = _dynamic_mem_mgr->allocate(ind, capacity);

    tensor->setBuffer(alloc);
  };

  if (tensor->buffer() == nullptr)
  {
    allocTensorMem();
  }
  // when buffer was already allocated and new_shape requires different size
  else if (tensor->total_size() != new_shape.num_elements() * sizeOfDataType(tensor->data_type()))
  {
    _dynamic_mem_mgr->deallocate(ind);

    allocTensorMem();
  }
  // when buffer with same size was already allocated, do nothing
}

void DynamicTensorManager::buildTensor(const ir::OperandIndex &ind,
                                       const ir::OperandInfo &tensor_info,
                                       ir::Layout backend_layout)
{
  assert(_tensors->find(ind) == _tensors->end());
  auto tensor = std::make_shared<operand::Tensor>(tensor_info, backend_layout);
  (*_tensors)[ind] = tensor;
}

void DynamicTensorManager::changeShape(const ir::OperandIndex &ind, const ir::Shape &new_shape)
{
  auto tensor = (*_tensors)[ind];
  assert(tensor);

  setShape(tensor.get(), new_shape);
  // once the shape is changed, the output of operations using this tensor should be re-calculated
  tensor->set_dynamic();
}

void DynamicTensorManager::planDealloc(ir::OperationIndex op_ind, ir::OperandIndex operand_ind)
{
  auto find = _dealloc_tensor_map.find(op_ind);
  if (find != _dealloc_tensor_map.end())
  {
    auto &input_set = find->second;
    input_set.emplace(operand_ind);
  }
  else
  {
    _dealloc_tensor_map.emplace(
        std::make_pair(op_ind, std::unordered_set<ir::OperandIndex>{operand_ind}));
  }
}

void DynamicTensorManager::deallocInput(ir::OperationIndex op_ind)
{
  auto find = _dealloc_tensor_map.find(op_ind);
  if (find == _dealloc_tensor_map.end())
    return;

  auto &input_set = find->second;
  for (auto input_ind : input_set)
  {
    if (!_tensors->at(input_ind)->is_dynamic())
      continue;

    _dynamic_mem_mgr->deallocate(input_ind);
    VERBOSE(DynamicTensorManager) << "Deallocating #" << input_ind.value()
                                  << " (input of op_ind: " << op_ind.value() << ")" << std::endl;
  }
}

void DynamicTensorManager::deallocSubgraphOutput(ir::OperandIndex output_ind)
{
  if (!_tensors->at(output_ind)->is_dynamic())
    return;

  _dynamic_mem_mgr->deallocate(output_ind);
  VERBOSE(DynamicTensorManager) << "Deallocating #" << output_ind.value()
                                << " (output of a subgraph)" << std::endl;
}

} // namespace controlflow
} // namespace backend
} // namespace onert
