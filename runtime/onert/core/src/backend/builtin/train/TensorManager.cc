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
namespace builtin
{
namespace train
{

TensorManager::TensorManager(const std::shared_ptr<TensorRegistry> &reg)
  : _nonconst_mgr{new MemoryManager()}, _tensors{reg->base_reg()}
{
  // DO NOTHING
}

void TensorManager::allocateNonConstTensors()
{
  _nonconst_mgr->allocate();

  // const auto &reg = _tensors->base_reg();
  for (auto &&pair : _tensors->nonconst_tensors())
  {
    const auto &index = pair.first;
    auto tensor = pair.second.get();
    assert(!tensor->is_dynamic());

    auto *buffer = _nonconst_mgr->getBuffer(index);
    tensor->setBuffer(buffer);
    VERBOSE(TensorManager) << "          TENSOR " << index << " : " << static_cast<void *>(buffer)
                           << std::endl;
  }
}

void TensorManager::claimNonConstPlan(const ir::OperandIndex &index)
{
  // const auto &reg = _tensors->base_reg();
  auto tensor = _tensors->getNonConstTensor(index);
  assert(tensor && !tensor->is_dynamic());

  _nonconst_mgr->claimPlan(index, tensor->total_size());
}

void TensorManager::releaseNonConstPlan(const ir::OperandIndex &index)
{
  assert(_tensors->getNonConstTensor(index) && !_tensors->getNonConstTensor(index)->is_dynamic());

  _nonconst_mgr->releasePlan(index);
}

} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
