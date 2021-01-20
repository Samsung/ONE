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

#ifndef __ONERT_BACKEND_ACL_COMMON_TENSOR_MANAGER_H__
#define __ONERT_BACKEND_ACL_COMMON_TENSOR_MANAGER_H__

#include <arm_compute/runtime/IMemoryManager.h>

#include "AclMemoryManager.h"
#include "AclInternalBufferManager.h"
#include "ir/OperandIndexMap.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor> class AclTensorManager
{
public:
  using T_AclMemoryManager = AclMemoryManager<T_ITensor, T_Tensor, T_SubTensor>;

  AclTensorManager(T_AclMemoryManager *const_mgr, T_AclMemoryManager *nonconst_mgr,
                   IInternalBufferManager *inter_mgr);

  virtual ~AclTensorManager() = default;

  void allocateConsts(void);
  void allocateNonconsts(void);
  void deallocateConsts(void);
  void deallocateNonconsts(void);

  void allocateInternalBufferManager(void);
  void deallocateInternalBufferManager(void);

  void buildTensor(const ir::OperandIndex &ind, const ::arm_compute::TensorInfo &info, size_t rank,
                   bool as_const, size_t num_uses);
  void buildSubtensor(const ir::OperandIndex &parent, const ir::OperandIndex &child,
                      const ::arm_compute::TensorShape &shape,
                      const ::arm_compute::Coordinates &coordinates, size_t rank,
                      bool extent_parent);

  std::shared_ptr<T_ITensor> findTensorAsParent(const ir::OperandIndex &ind);

  void startLifetime(const ir::OperandIndex &ind);
  void finishLifetime(const ir::OperandIndex &ind);

  std::shared_ptr<T_ITensor> at(const ir::OperandIndex &ind);

  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &constTensors(void);
  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &nonconstTensors(void);
  ir::OperandIndexMap<std::shared_ptr<T_SubTensor>> &nonconstSubtensors(void);

  std::shared_ptr<::arm_compute::IMemoryManager> internal_buffer_manager(void);

  void iterate(const std::function<void(const ir::OperandIndex &)> &fn);

  void tryDeallocConstants(void);

private:
  std::unique_ptr<T_AclMemoryManager> _const_mgr;
  std::unique_ptr<T_AclMemoryManager> _nonconst_mgr;
  std::unique_ptr<IInternalBufferManager> _inter_mgr;
  ir::OperandIndexMap<T_AclMemoryManager &> _ind_to_mgr;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#include <cassert>
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::AclTensorManager(
  T_AclMemoryManager *const_mgr, T_AclMemoryManager *nonconst_mgr,
  IInternalBufferManager *inter_mgr)
  : _const_mgr{const_mgr}, _nonconst_mgr{nonconst_mgr}, _inter_mgr{inter_mgr}
{
  // DO NOTHING
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::allocateConsts(void)
{
  VERBOSE(TENSOR_MANAGER_ACL) << "ALLOCATE CONSTS" << std::endl;
  _const_mgr->allocate();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::allocateNonconsts(void)
{
  _nonconst_mgr->allocate();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::deallocateConsts(void)
{
  _const_mgr->deallocate();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::deallocateNonconsts(void)
{
  _nonconst_mgr->deallocate();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::allocateInternalBufferManager(void)
{
  _inter_mgr->allocate();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::deallocateInternalBufferManager(void)
{
  _inter_mgr->deallocate();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::buildTensor(
  const ir::OperandIndex &ind, const ::arm_compute::TensorInfo &info, size_t rank, bool as_const,
  size_t num_uses)
{
  assert(_ind_to_mgr.find(ind) == _ind_to_mgr.end());
  if (as_const)
  {
    _const_mgr->buildTensor(ind, info, rank, num_uses);
    _ind_to_mgr.insert({ind, *_const_mgr});
  }
  else
  {
    _nonconst_mgr->buildTensor(ind, info, rank, num_uses);
    _ind_to_mgr.insert({ind, *_nonconst_mgr});
  }
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::buildSubtensor(
  const ir::OperandIndex &parent, const ir::OperandIndex &child,
  const ::arm_compute::TensorShape &shape, const ::arm_compute::Coordinates &coordinates,
  size_t rank, bool extent_parent)
{
  assert(_ind_to_mgr.find(child) == _ind_to_mgr.end());
  std::shared_ptr<T_ITensor> parent_tensor = findTensorAsParent(parent);
  assert(parent_tensor);
  _nonconst_mgr->buildSubtensor(parent_tensor, child, shape, coordinates, rank, extent_parent);
  _ind_to_mgr.insert({child, *_nonconst_mgr});
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
std::shared_ptr<T_ITensor>
AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::findTensorAsParent(const ir::OperandIndex &ind)
{

  auto &tensors = _nonconst_mgr->tensors();
  auto &subtensors = _nonconst_mgr->subtensors();
  if (tensors.find(ind) != tensors.end())
  {
    // Parent is allocated as tensor
    return tensors[ind];
  }
  else if (subtensors.find(ind) != subtensors.end())
  {
    // Parent is allocated as subtensor
    return subtensors[ind];
  }
  else
  {
    return nullptr;
  }
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::startLifetime(const ir::OperandIndex &ind)
{
  assert(_ind_to_mgr.find(ind) != _ind_to_mgr.end());
  _ind_to_mgr.at(ind).startLifetime(ind);
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::finishLifetime(const ir::OperandIndex &ind)
{
  assert(_ind_to_mgr.find(ind) != _ind_to_mgr.end());
  _ind_to_mgr.at(ind).finishLifetime(ind);
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
std::shared_ptr<T_ITensor>
AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::at(const ir::OperandIndex &ind)
{
  if (_ind_to_mgr.find(ind) == _ind_to_mgr.end())
    return nullptr;

  auto &tensors = _ind_to_mgr.at(ind).tensors();
  if (tensors.find(ind) != tensors.end())
  {
    return tensors.at(ind);
  }
  else
  {
    auto subtensors = _ind_to_mgr.at(ind).subtensors();
    auto itr = subtensors.find(ind);
    if (itr == subtensors.end())
      return nullptr;
    else
      return itr->second;
  }
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &
AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::constTensors(void)
{
  return _const_mgr->tensors();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &
AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::nonconstTensors(void)
{
  return _nonconst_mgr->tensors();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
ir::OperandIndexMap<std::shared_ptr<T_SubTensor>> &
AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::nonconstSubtensors(void)
{
  return _nonconst_mgr->subtensors();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
std::shared_ptr<::arm_compute::IMemoryManager>
AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::internal_buffer_manager(void)
{
  return _inter_mgr->internal_buffer_manager();
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::iterate(
  const std::function<void(const ir::OperandIndex &)> &fn)
{
  for (auto it : _nonconst_mgr->tensors())
    fn(it.first);

  for (auto it : _nonconst_mgr->subtensors())
    fn(it.first);

  for (auto it : _const_mgr->tensors())
    fn(it.first);
}

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor>
void AclTensorManager<T_ITensor, T_Tensor, T_SubTensor>::tryDeallocConstants(void)
{
  auto &tensors = _const_mgr->tensors();

  for (auto it = tensors.begin(); it != tensors.end();)
  {
    const auto &ind = it->first;
    auto tensor = it->second;
    // NOTE The condition "tensor->num_uses() < 2" is used to prevent deallocating a constant tensor
    // used in several nodes.
    if (tensor->handle() && !tensor->handle()->is_used() && tensor->num_uses() < 2)
    {
      VERBOSE(AclTensorManager) << "Tensor " << ind
                                << " will be deallocated as an unused constant tensor" << std::endl;
      tensor->allocator()->free();
      tensor.reset();
      it = tensors.erase(it);
    }
    else
    {
      ++it;
    }
  }
}

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_TENSOR_MANAGER_H__
