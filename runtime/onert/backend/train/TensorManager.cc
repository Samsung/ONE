/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
namespace train
{

TensorManager::TensorManager(const std::shared_ptr<TensorRegistry> &reg,
                             const std::string planner_id)
  : _nonconst_mgr{new MemoryManager(planner_id)},
    _trainable_mgr{new MemoryManager(planner_id)}, _tensors{reg}
{
  // DO NOTHING
}

void TensorManager::allocateNonconsts()
{
  _nonconst_mgr->allocate();

  for (auto &&pair : _tensors->nonconst_tensors())
  {
    const auto &ind = pair.first;
    auto tensor = pair.second.get();
    assert(!tensor->is_dynamic());
    if (!_as_constants[ind])
    {
      auto *buffer = _nonconst_mgr->getBuffer(ind);
      tensor->setBuffer(buffer);

      VERBOSE(TensorManager) << "          TENSOR " << ind << " : " << static_cast<void *>(buffer)
                             << std::endl;
    }
  }
}

void TensorManager::allocateTrainableTensors()
{
  _trainable_mgr->allocate();

  for (auto &&pair : _tensors->trainable_tensors())
  {
    const auto &ind = pair.first;
    auto tensor = pair.second.get();
    assert(!tensor->is_dynamic());
    if (_as_constants[ind])
    {
      auto *buffer = _trainable_mgr->getBuffer(ind);
      tensor->setBuffer(buffer);

      VERBOSE(TensorManager) << "TRAINABLE TENSOR " << ind << " : " << static_cast<void *>(buffer)
                             << std::endl;
    }
  }
}

void TensorManager::deallocateNonconsts()
{
  _nonconst_mgr->deallocate();
  _trainable_mgr->deallocate();
}

void TensorManager::buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &tensor_info,
                                ir::Layout backend_layout, bool as_const)
{
  assert(!_tensors->getNonConstTensor(ind) && !_tensors->getTrainableTensor(ind));

  if (!as_const)
  {
    auto tensor = std::make_unique<Tensor>(tensor_info, backend_layout);
    _tensors->setNonConstTensor(ind, std::move(tensor));
  }
  else
  {
    auto trainable_tensor = std::make_unique<TrainableTensor>(tensor_info, backend_layout);
    _tensors->setTrainableTensor(ind, std::move(trainable_tensor));
  }

  _as_constants[ind] = as_const;
}

void TensorManager::claimPlan(const ir::OperandIndex &ind, uint32_t size)
{
  assert(_tensors->getNativeITensor(ind) && !_tensors->getNativeITensor(ind)->is_dynamic());

  if (!_as_constants[ind])
    _nonconst_mgr->claimPlan(ind, size);
  else
    _trainable_mgr->claimPlan(ind, size);
}

void TensorManager::releasePlan(const ir::OperandIndex &ind)
{
  assert(_tensors->getNativeITensor(ind) && !_tensors->getNativeITensor(ind)->is_dynamic());

  if (!_as_constants[ind])
    _nonconst_mgr->releasePlan(ind);
  else
    _trainable_mgr->releasePlan(ind);
}

} // namespace train
} // namespace backend
} // namespace onert
