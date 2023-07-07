/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cassert>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

TensorManager::TensorManager(MemoryManager *const_mgr, MemoryManager *nonconst_mgr)
  : _const_mgr{const_mgr}, _nonconst_mgr{nonconst_mgr}
{
  // DO NOTHING
}

void TensorManager::allocateConsts(void) { _const_mgr->allocate(); }

void TensorManager::allocateNonconsts(void) { _nonconst_mgr->allocate(); }

void TensorManager::deallocateConsts(void) { _const_mgr->deallocate(); }

void TensorManager::deallocateNonconsts(void) { _nonconst_mgr->deallocate(); }

void TensorManager::buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                                TensorType type)
{
  assert(_ind_to_mgr.find(ind) == _ind_to_mgr.end());

  if (info.isConstant())
  {
    _const_mgr->buildTensor(ind, info, type);
    _ind_to_mgr.insert({ind, *_const_mgr});
  }
  else
  {
    _nonconst_mgr->buildTensor(ind, info, type);
    _ind_to_mgr.insert({ind, *_nonconst_mgr});
  }
}
ir::OperandIndex TensorManager::addTensor(const ir::Shape &shape)
{
  auto ind = _nonconst_mgr->addTensor(shape);
  _ind_to_mgr.insert({ind, *_nonconst_mgr});

  return ind;
}

void TensorManager::startLifetime(const ir::OperandIndex &ind)
{
  assert(_ind_to_mgr.find(ind) != _ind_to_mgr.end());
  _ind_to_mgr.at(ind).startLifetime(ind);
}

void TensorManager::finishLifetime(const ir::OperandIndex &ind)
{
  assert(_ind_to_mgr.find(ind) != _ind_to_mgr.end());
  _ind_to_mgr.at(ind).finishLifetime(ind);
}

std::shared_ptr<operand::ICLTensor> TensorManager::at(const ir::OperandIndex &ind)
{
  if (_ind_to_mgr.find(ind) == _ind_to_mgr.end())
    return nullptr;

  auto &tensors = _ind_to_mgr.at(ind).tensors();
  if (tensors.find(ind) != tensors.end())
  {
    return tensors.at(ind);
  }

  return nullptr;
}

ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> &TensorManager::constTensors(void)
{
  return _const_mgr->tensors();
}

ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> &TensorManager::nonconstTensors(void)
{
  return _nonconst_mgr->tensors();
}

void TensorManager::iterate(const std::function<void(const ir::OperandIndex &)> &fn)
{
  for (auto &it : _nonconst_mgr->tensors())
    fn(it.first);

  for (auto &it : _const_mgr->tensors())
    fn(it.first);
}

void TensorManager::tryDeallocConstants(void)
{
  // NYI
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
