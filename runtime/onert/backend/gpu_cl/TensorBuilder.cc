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
#include "ParentInfo.h"

#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
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

TensorBuilder::TensorBuilder(const ir::Operands &operands, TensorManager *tensor_mgr,
                             tflite::gpu::cl::InferenceContext::CreateInferenceInfo create_info,
                             const std::shared_ptr<tflite::gpu::cl::Environment> &environment)
  : _operands{operands}, _tensor_mgr{tensor_mgr}, _create_info{create_info}, _environment{
                                                                               environment}
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
  // Update lifetime sequence to apply subtensor optimization

  std::unordered_map<ir::OperandIndex, ir::OperandIndex> root_map;
  std::function<ir::OperandIndex &(ir::OperandIndex)> find_root =
    [&](ir::OperandIndex ind) -> ir::OperandIndex & {
    ir::OperandIndex &ret = root_map[ind];

    // We know the root parent value already
    if (ret.valid())
      return ret;

    auto itr = _parent_map.find(ind);
    if (itr == _parent_map.end())
    {
      // If there is no parent, let's store the value of itself
      return ret = ind;
    }
    else
    {
      return ret = find_root(itr->second.parent);
    }
  };

  ir::OperandIndexMap<bool> first_use_check;
  ir::OperandIndexMap<bool> last_use_check;
  std::map<size_t, std::pair<UsesType, ir::OperandIndex>> lifetime_map;
  for (size_t i = 0; i < _lifetime_seq.size(); i++)
  {
    auto &entry = _lifetime_seq[i];
    if (entry.first != UsesType::FIRST)
      continue;
    auto root_ind = find_root(entry.second);
    if (first_use_check[root_ind])
      continue;
    first_use_check[root_ind] = true;
    lifetime_map[i] = {UsesType::FIRST, root_ind};
  }

  for (int i = _lifetime_seq.size() - 1; i >= 0; i--)
  {
    auto &entry = _lifetime_seq[i];
    if (entry.first != UsesType::LAST)
      continue;
    auto root_ind = find_root(entry.second);
    if (last_use_check[root_ind])
      continue;
    last_use_check[root_ind] = true;
    lifetime_map[i] = {UsesType::LAST, root_ind};
  }

  for (auto &entry : lifetime_map)
  {
    auto &use = entry.second;
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
  for (auto &entry : _tensor_info_map)
  {
    auto ind = entry.first;
    if (_parent_map.count(ind) > 0)
      continue;
    auto type = _tensor_type_map.at(ind);
    const auto &info = entry.second;
    _tensor_mgr->buildTensor(ind, info, _create_info, _environment, _environment->device().info_,
                             type);
  }
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
