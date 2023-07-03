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

namespace
{

using namespace onert;
using namespace onert::backend::train;

template <typename Tensor>
void allocateMemory(MemoryManager *mgr, const ir::OperandIndexMap<std::unique_ptr<Tensor>> &tensors,
                    std::string tensor_type)
{
  mgr->allocate();

  for (auto &&pair : tensors)
  {
    const auto &index = pair.first;
    auto tensor = pair.second.get();
    assert(!tensor->is_dynamic());

    auto *buffer = mgr->getBuffer(index);
    tensor->setBuffer(buffer);
    VERBOSE(TensorManager) << tensor_type << index << " : " << static_cast<void *>(buffer)
                           << std::endl;
  }
}

} // namespace

namespace onert
{
namespace backend
{
namespace train
{

TensorManager::TensorManager(const std::shared_ptr<TensorRegistry> &reg,
                             const std::string planner_id)
  : _nonconst_mgr{new MemoryManager(planner_id)}, _trainable_mgr{new MemoryManager(planner_id)},
    _derivative_mgr{new MemoryManager(planner_id)},
    _gradient_mgr{new MemoryManager(planner_id)}, _tensors{reg}
{
  // DO NOTHING
}

void TensorManager::allocateForwardTensors()
{
  allocateMemory(_nonconst_mgr.get(), _tensors->nonconst_tensors(),
                 std::string{"          TENSOR "});
  allocateMemory(_trainable_mgr.get(), _tensors->trainable_tensors(),
                 std::string{"TRAINABLE TENSOR "});
}

void TensorManager::allocateBackwardTensors()
{
  allocateMemory(_derivative_mgr.get(), _tensors->derivative_tensors(),
                 std::string{"DERIVATIVE TENSOR "});
  allocateMemory(_gradient_mgr.get(), _tensors->gradient_tensors(),
                 std::string{"GRADIENT TENSOR "});
}

void TensorManager::buildForwardTensor(const ir::OperandIndex &index,
                                       const ir::OperandInfo &tensor_info,
                                       ir::Layout backend_layout, bool as_const)
{
  assert(!_tensors->getNonConstTensor(index) && !_tensors->getTrainableTensor(index));

  if (!as_const)
  {
    auto tensor = std::make_unique<Tensor>(tensor_info, backend_layout);
    _tensors->setNonConstTensor(index, std::move(tensor));
  }
  else
  {
    auto tensor = std::make_unique<TrainableTensor>(tensor_info, backend_layout);
    _tensors->setTrainableTensor(index, std::move(tensor));
  }
  _as_constants[index] = as_const;
}

void TensorManager::buildBackwardTensor(const ir::OperandIndex &index,
                                        const ir::OperandInfo &tensor_info,
                                        ir::Layout backend_layout, bool as_const)
{
  assert(!_tensors->getDerivativeTensor(index) && !_tensors->getGradientTensor(index));
  assert(_as_constants[index] == as_const);

  if (!as_const)
  {
    auto tensor = std::make_unique<DerivativeTensor>(tensor_info, backend_layout);
    _tensors->setDerivativeTensor(index, std::move(tensor));
  }
  else
  {
    auto tensor = std::make_unique<GradientTensor>(tensor_info, backend_layout);
    _tensors->setGradientTensor(index, std::move(tensor));
  }
}

void TensorManager::claimForwardPlan(const ir::OperandIndex &index)
{
  auto tensor = _tensors->getNativeITensor(index);
  assert(tensor && tensor->is_dynamic());

  auto size = tensor->total_size();
  if (!_as_constants[index])
    _nonconst_mgr->claimPlan(index, size);
  else
    _trainable_mgr->claimPlan(index, size);

  // TODO Consider derivative and gradient tensors
}

void TensorManager::releaseForwardPlan(const ir::OperandIndex &index)
{
  assert(_tensors->getNativeITensor(index) && !_tensors->getNativeITensor(index)->is_dynamic());

  if (!_as_constants[index])
    _nonconst_mgr->releasePlan(index);

  // TODO Support the release plan for trainable tensors
}

void TensorManager::claimBackwardPlan(const ir::OperandIndex &index)
{
  ITensor *tensor = _tensors->getDerivativeTensor(index);
  if (tensor == nullptr)
    tensor = _tensors->getGradientTensor(index);
  assert(tensor && !tensor->is_dynamic());

  auto size = tensor->total_size();
  if (!_as_constants[index])
    _derivative_mgr->claimPlan(index, size);
  else
    _gradient_mgr->claimPlan(index, size);
}

void TensorManager::releaseBackwardPlan(const ir::OperandIndex &index)
{
  ITensor *tensor = _tensors->getDerivativeTensor(index);
  if (tensor == nullptr)
    tensor = _tensors->getGradientTensor(index);
  assert(tensor && !tensor->is_dynamic());

  if (!_as_constants[index])
    _derivative_mgr->releasePlan(index);
  else
    _gradient_mgr->releasePlan(index);

  // TODO Consider non-const and trainable tensors
}

} // namespace train
} // namespace backend
} // namespace onert
