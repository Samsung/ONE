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

#ifndef __ONERT_BACKEND_ACL_COMMON_MEMORY_MANAGER_H__
#define __ONERT_BACKEND_ACL_COMMON_MEMORY_MANAGER_H__

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/IMemoryManager.h>
#include <cassert>

#include "ir/OperandIndexMap.h"
#include "Convert.h"
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor> class AclMemoryManager
{
public:
  AclMemoryManager()
  {
    // DO NOTHING
  }

  virtual ~AclMemoryManager() = default;

  virtual void allocate(void)
  {
    for (const auto &tensor_entry : _tensors)
    {
      auto tensor = tensor_entry.second;
      VERBOSE(ALLOCATE_TENSOR_ACL) << tensor_entry.first << std::endl;
      tensor->allocator()->allocate();
    }
  }

  virtual void deallocate(void)
  {
    for (const auto &tensor_entry : _tensors)
    {
      auto tensor = tensor_entry.second;
      tensor->allocator()->free();
    }
  }

  virtual void startLifetime(const ir::OperandIndex &)
  { /* DO NOTHING */
  }
  virtual void finishLifetime(const ir::OperandIndex &)
  { /* DO NOTHING */
  }

  void buildTensor(const ir::OperandIndex &ind, const ::arm_compute::TensorInfo &info, size_t rank,
                   size_t num_uses)
  {
    auto tensor = std::make_shared<T_Tensor>(info, rank, num_uses);
    _tensors[ind] = tensor;
  }

  void buildSubtensor(std::shared_ptr<T_ITensor> parent_tensor, const ir::OperandIndex &child_ind,
                      const ::arm_compute::TensorShape &shape,
                      const ::arm_compute::Coordinates &coordinates, size_t rank,
                      bool extent_parent)
  {
    auto subtensor =
      std::make_shared<T_SubTensor>(parent_tensor.get(), shape, coordinates, rank, extent_parent);
    _subtensors[child_ind] = subtensor;
  }

  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &tensors(void) { return _tensors; }

  ir::OperandIndexMap<std::shared_ptr<T_SubTensor>> &subtensors(void) { return _subtensors; }

private:
  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> _tensors;
  ir::OperandIndexMap<std::shared_ptr<T_SubTensor>> _subtensors;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_MEMORY_MANAGER_H__
