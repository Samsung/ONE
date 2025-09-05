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

#include "backend/basic/StaticTensorManager.h"

#include "backend/basic/DynamicTensorManager.h"
#include "backend/basic/Tensor.h"

#include <util/logging.h>

#include <algorithm>

namespace onert::backend::basic
{

StaticTensorManager::StaticTensorManager(
  const std::shared_ptr<TensorRegistry> &reg, DynamicTensorManager *dynamic_tensor_manager,
  const ir::OperandIndexMap<ir::OperandIndex> &shared_memory_operand_indexes)
  : _nonconst_mgr{new MemoryManager()}, _tensors{reg},
    _dynamic_tensor_manager{dynamic_tensor_manager},
    _shared_memory_operand_indexes{shared_memory_operand_indexes}
{
  // DO NOTHING
}

StaticTensorManager::StaticTensorManager(
  const std::shared_ptr<TensorRegistry> &reg, const std::string planner_id,
  DynamicTensorManager *dynamic_tensor_manager,
  const ir::OperandIndexMap<ir::OperandIndex> &shared_memory_operand_indexes)
  : _nonconst_mgr{new MemoryManager(planner_id)}, _tensors{reg},
    _dynamic_tensor_manager{dynamic_tensor_manager},
    _shared_memory_operand_indexes{shared_memory_operand_indexes}
{
  // DO NOTHING
}

void StaticTensorManager::allocateNonconsts(void)
{
  _nonconst_mgr->allocate();

  for (auto &&[ind, tensor] : _tensors->native_tensors())
  {
    const auto adjusted_ind = adjustWithMemorySourceOperand(ind);
    if (!_as_constants[adjusted_ind] && !tensor->is_dynamic())
    {
      auto *buffer = _nonconst_mgr->getBuffer(adjusted_ind);
      tensor->setBuffer(buffer);

      VERBOSE(CPU_StaticTensorManager)
        << "TENSOR " << ind << " : " << static_cast<void *>(buffer) << std::endl;
    }
  }
}

void StaticTensorManager::deallocateNonconsts(void) { _nonconst_mgr->deallocate(); }

void StaticTensorManager::buildTensor(const ir::OperandIndex &ind,
                                      const ir::OperandInfo &tensor_info, bool as_const)
{
  assert(!_tensors->getNativeTensor(ind));
  if (as_const)
  {
    auto tensor = std::make_unique<ExternalTensor>(tensor_info);
    _tensors->setNativeTensor(ind, std::move(tensor));
  }
  else
  {
    auto tensor =
      std::make_unique<Tensor>(tensor_info, _dynamic_tensor_manager->dynamic_mem_mgr().get());
    _tensors->setNativeTensor(ind, std::move(tensor));
  }
  _as_constants[ind] = as_const;
}

void StaticTensorManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  assert(_tensors->getNativeTensor(ind));

  // This method is called only when a tensor has proper shape
  assert(!_tensors->getNativeTensor(ind)->is_dynamic());

  const auto claim_ind = adjustWithMemorySourceOperand(ind);
  if (_as_constants[claim_ind])
  {
    return;
  }
  if (isSharedMemoryOperand(claim_ind))
  {
    ++_source_operand_inds_ref_counter[claim_ind];
    if (_source_operand_inds_ref_counter[claim_ind] > 1)
    {
      return; // claimPlan should be called only for the first usage
    }
  }
  _nonconst_mgr->claimPlan(claim_ind, size);
}

void StaticTensorManager::releasePlan(const ir::OperandIndex &ind)
{
  assert(_tensors->getNativeTensor(ind));

  // This method is called only when a tensor has proper shape
  assert(!_tensors->getNativeTensor(ind)->is_dynamic());

  const auto release_ind = adjustWithMemorySourceOperand(ind);
  if (_as_constants[release_ind])
  {
    return;
  }
  if (isSharedMemoryOperand(release_ind))
  {
    if (_source_operand_inds_ref_counter[release_ind] > 0) // sanity check
    {
      --_source_operand_inds_ref_counter[release_ind];
    }
    if (_source_operand_inds_ref_counter[release_ind] > 0)
    {
      return; // releasePlan should be called only for the first usage
    }
  }
  _nonconst_mgr->releasePlan(release_ind);
}

void StaticTensorManager::iterate(const std::function<void(const ir::OperandIndex &)> &fn)
{
  for (const auto &it : _tensors->native_tensors())
    fn(it.first);
}

ir::OperandIndex
StaticTensorManager::adjustWithMemorySourceOperand(const ir::OperandIndex &ind) const
{
  const auto shared_operand_ind = _shared_memory_operand_indexes.find(ind);
  if (shared_operand_ind != std::end(_shared_memory_operand_indexes))
  {
    return shared_operand_ind->second;
  }
  // source memory operand not found
  return ind;
}

bool StaticTensorManager::isSharedMemoryOperand(const ir::OperandIndex &ind) const
{
  for (const auto &[shared_ind, source_ind] : _shared_memory_operand_indexes)
  {
    if (shared_ind == ind || source_ind == ind)
    {
      return true;
    }
  }
  return false;
}

} // namespace onert::backend::basic
