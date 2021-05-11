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

#ifndef __ONERT_BACKEND_CL_TENSOR_BUILDER_H__
#define __ONERT_BACKEND_CL_TENSOR_BUILDER_H__

#include <memory>
#include <queue>

#include "ClTensorManager.h"
#include "ClTensorRegistry.h"
#include "ParentInfo.h"

#include "open_cl/TensorType.h"
#include "open_cl/TensorTypeUtil.h"
#include "open_cl/ClDevice.h"
#include "open_cl/InferenceContext.h"

#include "ir/OperandIndexMap.h"
#include "ir/OperandIndexSequence.h"
#include <ir/Operands.h>
#include <util/Utils.h>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

enum class UsesType
{
  FIRST,
  LAST
};

template <typename T_ITensor, typename T_Tensor> class ClTensorBuilder
{
public:
  using T_ClTensorManager = ClTensorManager<T_ITensor, T_Tensor>;

  ClTensorBuilder(const ir::Operands &operands, T_ClTensorManager *tensor_mgr,
                  InferenceContext::CreateInferenceInfo create_info, CLCommandQueue *queue,
                  CLDevice *device);

  /**
   * @brief     Register tensor information to allocate on ACL-CL backend
   * @param[in] ind    Operand index
   * @param[in] info   Tensor information
   * @param[in] layout Tensor data layout
   */
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                          ir::Layout backend_layout);

  void notifyFirstUse(const ir::OperandIndex &);
  void notifyLastUse(const ir::OperandIndex &);

  bool isRegistered(const ir::OperandIndex &) const;

  void prepare();
  void allocate();
  void postFunctionPrepare();

  T_ClTensorManager *cl_tensor_manager(void) { return _tensor_mgr.get(); }

  void setUsesCount(const ir::OperandIndex &index, size_t num_uses)
  {
    assert(_uses_count_map.find(index) != _uses_count_map.end() ? _uses_count_map[index] == num_uses
                                                                : true);
    _uses_count_map[index] = num_uses;
  }

  void parent_map(std::unordered_map<ir::OperandIndex, ParentInfo> &&parent_map)
  {
    _parent_map = std::move(parent_map);
  }

  bool areSubTensorsOf(const ir::OperandIndex &parent, const ir::OperandIndexSequence &seq);

  /**
   * @brief     Check child tensor is allocated as subtensor of parent tensor
   * @param[in] parent  Index of parent
   * @param[in] child   Index of child
   * @return    @c true if child is allocated as subtensor of parent, otherwise @c false
   */
  bool isSubTensorOf(const ir::OperandIndex &parent, const ir::OperandIndex &child);

private:
  void buildTensors(void);
  ir::OperandIndex findRootParent(ir::OperandIndex index);

private:
  const ir::Operands &_operands;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
  ir::OperandIndexMap<ir::Layout> _tensor_layout_map;
  ir::OperandIndexMap<size_t> _uses_count_map;

  std::unique_ptr<T_ClTensorManager> _tensor_mgr;
  InferenceContext::CreateInferenceInfo _create_info;
  CLCommandQueue *_queue;
  CLDevice *_device;
  // for linear executor
  std::vector<std::pair<UsesType, ir::OperandIndex>> _lifetime_seq;

  // Extra info for concat elimination
  ir::OperandIndexMap<ParentInfo> _parent_map;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#include <cassert>
#include <stack>

#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

template <typename T_ITensor, typename T_Tensor>
ClTensorBuilder<T_ITensor, T_Tensor>::ClTensorBuilder(
  const ir::Operands &operands, T_ClTensorManager *tensor_mgr,
  InferenceContext::CreateInferenceInfo create_info, CLCommandQueue *queue, CLDevice *device)
  : _operands{operands}, _tensor_mgr{tensor_mgr},
    _create_info{create_info}, _queue{queue}, _device{device}
{
  assert(_tensor_mgr);
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorBuilder<T_ITensor, T_Tensor>::registerTensorInfo(const ir::OperandIndex &ind,
                                                              const ir::OperandInfo &info,
                                                              ir::Layout backend_layout)
{
  assert(_tensor_mgr->constTensors().size() == 0);
  assert(_tensor_mgr->nonconstTensors().size() == 0);

  _uses_count_map[ind] = _operands.at(ind).getUses().size();

  _tensor_info_map.emplace(ind, info);
  _tensor_layout_map.insert({ind, backend_layout});
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorBuilder<T_ITensor, T_Tensor>::notifyFirstUse(const ir::OperandIndex &ind)
{
  _lifetime_seq.emplace_back(UsesType::FIRST, ind);
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorBuilder<T_ITensor, T_Tensor>::notifyLastUse(const ir::OperandIndex &ind)
{
  _lifetime_seq.emplace_back(UsesType::LAST, ind);
}

template <typename T_ITensor, typename T_Tensor>
bool ClTensorBuilder<T_ITensor, T_Tensor>::isRegistered(const ir::OperandIndex &ind) const
{
  return _tensor_info_map.find(ind) != _tensor_info_map.end();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorBuilder<T_ITensor, T_Tensor>::prepare(void)
{
  buildTensors();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorBuilder<T_ITensor, T_Tensor>::allocate(void)
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

template <typename T_ITensor, typename T_Tensor>
void ClTensorBuilder<T_ITensor, T_Tensor>::postFunctionPrepare(void)
{
  _tensor_mgr->tryDeallocConstants();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorBuilder<T_ITensor, T_Tensor>::buildTensors(void)
{
  assert(_tensor_mgr->constTensors().size() == 0);
  assert(_tensor_mgr->nonconstTensors().size() == 0);
  // Normal tensors
  for (auto &entry : _tensor_info_map)
  {
    auto ind = entry.first;
    if (_parent_map.count(ind) > 0)
      continue;

    const auto &info = entry.second;
    _tensor_mgr->buildTensor(ind, info, _uses_count_map[ind], _create_info, _queue, _device->info_);
  }
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_TEMPL_TENSOR_BUILDER_H__
