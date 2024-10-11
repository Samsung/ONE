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

#include <algorithm>

#include "backend/basic/StaticTensorManager.h"

#include "backend/basic/DynamicTensorManager.h"
#include "backend/basic/Tensor.h"
#include <util/logging.h>

namespace onert
{
namespace backend
{
namespace basic
{

StaticTensorManager::StaticTensorManager(
  const std::shared_ptr<TensorRegistry> &reg, DynamicTensorManager *dynamic_tensor_manager,
  const ir::OperandIndexMap<ir::OperandIndex> &operands_with_shared_memory)
  : _nonconst_mgr{new MemoryManager()}, _tensors{reg},
    _dynamic_tensor_manager{dynamic_tensor_manager},
    _operands_with_shared_memory{operands_with_shared_memory}
{
  // DO NOTHING
}

StaticTensorManager::StaticTensorManager(
  const std::shared_ptr<TensorRegistry> &reg, const std::string planner_id,
  DynamicTensorManager *dynamic_tensor_manager,
  const ir::OperandIndexMap<ir::OperandIndex> &operands_with_shared_memory)
  : _nonconst_mgr{new MemoryManager(planner_id)}, _tensors{reg},
    _dynamic_tensor_manager{dynamic_tensor_manager},
    _operands_with_shared_memory{operands_with_shared_memory}
{
  // DO NOTHING
}

void StaticTensorManager::allocateNonconsts(void)
{
  _nonconst_mgr->allocate();

  for (auto &&[ind, tensor] : _tensors->native_tensors())
  {
    bool buffer_set = false;
    if (!tensor->is_dynamic())
    {
      if (_operands_with_shared_memory.find(ind) != std::end(_operands_with_shared_memory))
      {
        const auto &shared_memory_ind = _operands_with_shared_memory[ind];
        if (!_as_constants[shared_memory_ind])
        {
          tensor->setBuffer(_nonconst_mgr->getBuffer(shared_memory_ind));
          buffer_set = true;
        }
      }
      else if (!_as_constants[ind])
      {
        tensor->setBuffer(_nonconst_mgr->getBuffer(ind));
        buffer_set = true;
      }
      if (buffer_set)
      {
        VERBOSE(CPU_StaticTensorManager)
          << "TENSOR " << ind << " : " << static_cast<void *>(tensor->buffer()) << std::endl;
      }
    }
  }
}

void StaticTensorManager::deallocateNonconsts(void) { _nonconst_mgr->deallocate(); }

void StaticTensorManager::buildTensor(const ir::OperandIndex &ind,
                                      const ir::OperandInfo &tensor_info, bool as_const)
{
  assert(!_tensors->getNativeTensor(ind));
  std::unique_ptr<Tensor> tensor = nullptr;
  if (as_const)
  {
    tensor = std::make_unique<ExternalTensor>(tensor_info);
  }
  else
  {
    const auto source_operand = _operands_with_shared_memory.find(ind);
    if (source_operand != std::end(_operands_with_shared_memory) &&
        _as_constants[source_operand->second])
    {
      as_const = _as_constants[source_operand->second];
      auto new_tensor_info = tensor_info;
      new_tensor_info.setAsConstant();
      tensor = std::make_unique<ExternalTensor>(new_tensor_info);
    }
    else
    {
      tensor =
        std::make_unique<Tensor>(tensor_info, _dynamic_tensor_manager->dynamic_mem_mgr().get());
    }
  }
  assert(tensor);
  _tensors->setNativeTensor(ind, std::move(tensor));
  _as_constants[ind] = as_const;
}

void StaticTensorManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  assert(_tensors->getNativeTensor(ind));

  // This method is called only when a tensor has proper shape
  assert(!_tensors->getNativeTensor(ind)->is_dynamic());

  ir::OperandIndex claim_ind;
  const auto source_ind = _operands_with_shared_memory.find(ind);
  if (source_ind == std::end(_operands_with_shared_memory))
  {
    claim_ind = ind;
  }
  else
  {
    claim_ind = source_ind->second;
  }
  if (_as_constants[claim_ind])
  {
    return;
  }
  ++_source_operands_ref_counter[claim_ind];
  // notify only first usage
  if (1 == _source_operands_ref_counter[claim_ind]) {
    _nonconst_mgr->claimPlan(claim_ind, size);
  }
}

void StaticTensorManager::releasePlan(const ir::OperandIndex &ind)
{
  assert(_tensors->getNativeTensor(ind));

  // This method is called only when a tensor has proper shape
  assert(!_tensors->getNativeTensor(ind)->is_dynamic());

  ir::OperandIndex release_ind;
  const auto source_operand_ind = _operands_with_shared_memory.find(ind);
  if (source_operand_ind == std::end(_operands_with_shared_memory))
  {
    release_ind = ind;
  }
  else
  {
    release_ind = source_operand_ind->second;
  }
  if(_as_constants[release_ind]) {
    return;
  }
  if(_source_operands_ref_counter[release_ind] > 0) {
    --_source_operands_ref_counter[release_ind];
  }
  // notify only last usage
  if (0 == _source_operands_ref_counter[release_ind])
  {
    _nonconst_mgr->releasePlan(release_ind);
  }
}

void StaticTensorManager::iterate(const std::function<void(const ir::OperandIndex &)> &fn)
{
  for (const auto &it : _tensors->native_tensors())
    fn(it.first);
}

} // namespace basic
} // namespace backend
} // namespace onert
