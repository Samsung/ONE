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

#ifndef __ONERT_BACKEND_ACL_COMMON_TENSOR_MANAGER_H__
#define __ONERT_BACKEND_ACL_COMMON_TENSOR_MANAGER_H__

#include "ClMemoryManager.h"

#include "open_cl/InferenceContext.h"
#include "open_cl/TensorType.h"

#include "ir/OperandInfo.h"
#include "ir/OperandIndexMap.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

template <typename T_ITensor, typename T_Tensor> class ClTensorManager
{
public:
  using T_ClMemoryManager = ClMemoryManager<T_ITensor, T_Tensor>;

  ClTensorManager(T_ClMemoryManager *const_mgr, T_ClMemoryManager *nonconst_mgr);

  virtual ~ClTensorManager() = default;

  void allocateConsts(void);
  void allocateNonconsts(void);
  void deallocateConsts(void);
  void deallocateNonconsts(void);

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                   InferenceContext::CreateInferenceInfo create_info,
                   std::shared_ptr<Environment> environment, DeviceInfo &device_info);

  std::shared_ptr<T_ITensor> findTensorAsParent(const ir::OperandIndex &ind);

  void startLifetime(const ir::OperandIndex &ind);
  void finishLifetime(const ir::OperandIndex &ind);

  std::shared_ptr<T_ITensor> at(const ir::OperandIndex &ind);
  std::shared_ptr<InferenceContext::DummyTensor> atR(const ir::OperandIndex &ind);

  InferenceContext::TensorReserver &constTensorReservers(void);
  InferenceContext::TensorReserver &nonconstTensorReservers(void);

  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &constTensors(void);
  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &nonconstTensors(void);

  void iterate(const std::function<void(const ir::OperandIndex &)> &fn);

  void tryDeallocConstants(void);

private:
  std::unique_ptr<T_ClMemoryManager> _const_mgr;
  std::unique_ptr<T_ClMemoryManager> _nonconst_mgr;
  ir::OperandIndexMap<T_ClMemoryManager &> _ind_to_mgr;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#include <cassert>
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

template <typename T_ITensor, typename T_Tensor>
ClTensorManager<T_ITensor, T_Tensor>::ClTensorManager(T_ClMemoryManager *const_mgr,
                                                      T_ClMemoryManager *nonconst_mgr)
  : _const_mgr{const_mgr}, _nonconst_mgr{nonconst_mgr}
{
  // DO NOTHING
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::allocateConsts(void)
{
  _const_mgr->allocate();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::allocateNonconsts(void)
{
  _nonconst_mgr->allocate();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::deallocateConsts(void)
{
  _const_mgr->deallocate();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::deallocateNonconsts(void)
{
  _nonconst_mgr->deallocate();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::buildTensor(
  const ir::OperandIndex &ind, const ir::OperandInfo &info,
  InferenceContext::CreateInferenceInfo create_info, std::shared_ptr<Environment> environment,
  DeviceInfo &device_info)
{
  assert(_ind_to_mgr.find(ind) == _ind_to_mgr.end());

  if (info.isConstant())
  {
    _const_mgr->buildTensor(ind, info, create_info, environment, device_info);
    _ind_to_mgr.insert({ind, *_const_mgr});
  }
  else
  {
    _nonconst_mgr->buildTensor(ind, info, create_info, environment, device_info);
    _ind_to_mgr.insert({ind, *_nonconst_mgr});
  }
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::startLifetime(const ir::OperandIndex &ind)
{
  assert(_ind_to_mgr.find(ind) != _ind_to_mgr.end());
  _ind_to_mgr.at(ind).startLifetime(ind);
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::finishLifetime(const ir::OperandIndex &ind)
{
  assert(_ind_to_mgr.find(ind) != _ind_to_mgr.end());
  _ind_to_mgr.at(ind).finishLifetime(ind);
}

template <typename T_ITensor, typename T_Tensor>
std::shared_ptr<T_ITensor> ClTensorManager<T_ITensor, T_Tensor>::at(const ir::OperandIndex &ind)
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

template <typename T_ITensor, typename T_Tensor>
ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &
ClTensorManager<T_ITensor, T_Tensor>::constTensors(void)
{
  return _const_mgr->tensors();
}

template <typename T_ITensor, typename T_Tensor>
ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &
ClTensorManager<T_ITensor, T_Tensor>::nonconstTensors(void)
{
  return _nonconst_mgr->tensors();
}

template <typename T_ITensor, typename T_Tensor>
std::shared_ptr<InferenceContext::DummyTensor>
ClTensorManager<T_ITensor, T_Tensor>::atR(const ir::OperandIndex &ind)
{
  if (_nonconst_mgr->tensorReservers().HaveTensor(ind.value()))
  {
    return _nonconst_mgr->tensorReservers().Get(ind.value());
  }
  else if (_const_mgr->tensorReservers().HaveTensor(ind.value()))
  {
    return _const_mgr->tensorReservers().Get(ind.value());
  }
  return nullptr;
}

template <typename T_ITensor, typename T_Tensor>
InferenceContext::TensorReserver &ClTensorManager<T_ITensor, T_Tensor>::constTensorReservers(void)
{
  return _const_mgr->tensorReservers();
}

template <typename T_ITensor, typename T_Tensor>
InferenceContext::TensorReserver &
ClTensorManager<T_ITensor, T_Tensor>::nonconstTensorReservers(void)
{
  return _nonconst_mgr->tensorReservers();
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::iterate(
  const std::function<void(const ir::OperandIndex &)> &fn)
{
  for (auto it : _nonconst_mgr->tensors())
    fn(it.first);

  for (auto it : _const_mgr->tensors())
    fn(it.first);
}

template <typename T_ITensor, typename T_Tensor>
void ClTensorManager<T_ITensor, T_Tensor>::tryDeallocConstants(void)
{
  // NYI
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_TENSOR_MANAGER_H__
