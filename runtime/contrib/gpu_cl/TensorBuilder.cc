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

#include <memory>
#include <queue>

#include "TensorBuilder.h"

#include "TensorManager.h"

#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"

#include "ir/OperandIndexMap.h"
#include "ir/OperandIndexSequence.h"
#include <ir/Operands.h>
#include <util/Utils.h>

#include <cassert>
#include <stack>

#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

using UsesType = cl_common::UsesType;

TensorBuilder::TensorBuilder(const ir::Operands &operands, TensorManager *tensor_mgr)
  : _operands{operands}, _tensor_mgr{tensor_mgr}
{
  assert(_tensor_mgr);
}

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                                       ir::Layout backend_layout, TensorType type)
{
  assert(_tensor_mgr->constTensors().size() == 0);
  assert(_tensor_mgr->nonconstTensors().size() == 0);

  _uses_count_map[ind] = _operands.at(ind).getUses().size();

  _tensor_info_map.emplace(ind, info);
  _tensor_type_map.emplace(ind, type);

  _tensor_layout_map.insert({ind, backend_layout});
}

void TensorBuilder::notifyFirstUse(const ir::OperandIndex &ind)
{
  _lifetime_seq.emplace_back(UsesType::FIRST, ind);
}

void TensorBuilder::notifyLastUse(const ir::OperandIndex &ind)
{
  _lifetime_seq.emplace_back(UsesType::LAST, ind);
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &ind) const
{
  return _tensor_info_map.find(ind) != _tensor_info_map.end();
}

void TensorBuilder::prepare(void) { buildTensors(); }

void TensorBuilder::allocate(void)
{
  auto lifetime_map = cl_common::createLifetimeMap(_lifetime_seq, _parent_map);

  for (const auto &entry : lifetime_map)
  {
    const auto &use = entry.second;
    auto use_type = use.first;
    auto use_index = use.second;
    assert(use_index.valid());
    if (use_type == UsesType::FIRST)
      _tensor_mgr->startLifetime(use_index);
    else
      _tensor_mgr->finishLifetime(use_index);
  }

  _tensor_mgr->allocateConsts();

  // TODO Since `_parent_map` is filled for all Concat nodes even if the node this backend uses
  //      After refactoring BackendContext we can uncomment this
  // assert(_tensor_info_map.size() ==
  //       _tensor_mgr->nonconstTensors().size() + num of constants of _tensor_info_map +
  //       _parent_map.size());
  _tensor_mgr->allocateNonconsts();
}

void TensorBuilder::postFunctionPrepare(void) { _tensor_mgr->tryDeallocConstants(); }

void TensorBuilder::buildTensors(void)
{
  assert(_tensor_mgr->constTensors().size() == 0);
  assert(_tensor_mgr->nonconstTensors().size() == 0);
  // Normal tensors
  for (const auto &entry : _tensor_info_map)
  {
    const auto &ind = entry.first;
    if (_parent_map.count(ind) > 0)
      continue;
    auto type = _tensor_type_map.at(ind);
    const auto &info = entry.second;
    _tensor_mgr->buildTensor(ind, info, type);
  }
}

ir::OperandIndex TensorBuilder::addTensor(const ir::Shape &shape)
{
  return _tensor_mgr->addTensor(shape);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
