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

namespace onert
{
namespace backend
{
namespace controlflow
{

DynamicTensorManager::DynamicTensorManager(const std::shared_ptr<cpu_common::TensorRegistry> &reg,
                                           const std::shared_ptr<UserTensorRegistry> &user_reg)
    : _dynamic_mem_mgr{new cpu_common::DynamicMemoryManager()}, _tensors{reg},
      _user_tensors{user_reg}
{
  // DO NOTHING
}

void DynamicTensorManager::applyShape(const ir::OperandIndex &ind, const ir::Shape &new_shape)
{
  // NOTE Handle user tensors first
  auto user_tensor = _user_tensors->getNativeTensor(ind);
  if (user_tensor)
  {
    // User tensors cannot be reallocated.
    auto buffer_size = user_tensor->total_size();
    auto new_size = new_shape.num_elements() * sizeOfDataType(user_tensor->data_type());
    if (buffer_size < new_size)
      throw std::runtime_error{"ExecutorBase: output buffer size is less than output tensor size"};
    user_tensor->setShape(new_shape);
  }

  // NOTE Then handle native tensors
  auto tensor = _tensors->getNativeTensor(ind);
  assert(tensor);

  bool previously_dynamic = tensor->is_dynamic();

  auto allocTensorMem = [&](bool overwrite = false) {
    auto capacity = tensor->total_size();
    auto alloc = _dynamic_mem_mgr->allocate(ind, capacity);

    if (overwrite)
      tensor->overwriteBuffer(alloc);
    else
      tensor->setBuffer(alloc);
  };

  if (!previously_dynamic)
  {
    // TODO deallocate tensor->buffer()
    // issue is that staticTensorManager might have allocate this memory
    tensor->setShape(new_shape);
    tensor->set_dynamic();
    allocTensorMem(true);
  }
  else if (tensor->buffer() == nullptr)
  {
    tensor->setShape(new_shape);
    tensor->set_dynamic();
    allocTensorMem();
  }
  // when buffer was already allocated and new_shape requires different size
  else
  {
    auto previous_size = tensor->total_size();
    auto new_size = new_shape.num_elements() * sizeOfDataType(tensor->data_type());
    if (previous_size != new_size)
    {
      _dynamic_mem_mgr->deallocate(ind);

      tensor->setShape(new_shape);
      tensor->set_dynamic();
      allocTensorMem(true);
    }
    else
    { // when buffer with same size was already allocated, shape could differ
      tensor->setShape(new_shape);
    }
  }
}

void DynamicTensorManager::buildTensor(const ir::OperandIndex &ind,
                                       const ir::OperandInfo &tensor_info,
                                       ir::Layout backend_layout)
{
  assert(_tensors->getNativeTensor(ind) == nullptr);
  auto tensor = std::make_shared<cpu_common::Tensor>(tensor_info, backend_layout);
  _tensors->setNativeTensor(ind, tensor);
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
    if (!_tensors->getNativeTensor(input_ind)->is_dynamic())
      continue;

    _dynamic_mem_mgr->deallocate(input_ind);
    VERBOSE(DynamicTensorManager) << "Deallocating #" << input_ind.value()
                                  << " (input of op_ind: " << op_ind.value() << ")" << std::endl;
  }
}

void DynamicTensorManager::deallocSubgraphOutput(ir::OperandIndex output_ind)
{
  if (!_tensors->getNativeTensor(output_ind)->is_dynamic())
    return;

  _dynamic_mem_mgr->deallocate(output_ind);
  VERBOSE(DynamicTensorManager) << "Deallocating #" << output_ind.value()
                                << " (output of a subgraph)" << std::endl;
}

} // namespace controlflow
} // namespace backend
} // namespace onert
