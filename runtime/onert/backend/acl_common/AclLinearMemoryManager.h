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

#ifndef __ONERT_BACKEND_ACL_COMMON_LINEAR_MEMORY_MANAGER_H__
#define __ONERT_BACKEND_ACL_COMMON_LINEAR_MEMORY_MANAGER_H__

#include <cassert>

#include "AclMemoryManager.h"
#include "ir/OperandIndexMap.h"
#include "util/logging.h"

namespace
{

template <typename T_MemoryManager, typename T_PoolManager, typename T_LifetimeManager>
std::shared_ptr<T_MemoryManager> createMemoryManager()
{
  std::shared_ptr<T_LifetimeManager> lifetime_mgr = std::make_shared<T_LifetimeManager>();
  std::shared_ptr<T_PoolManager> pool_mgr = std::make_shared<T_PoolManager>();

  std::shared_ptr<T_MemoryManager> mem_mgr =
    std::make_shared<T_MemoryManager>(lifetime_mgr, pool_mgr);
  return mem_mgr;
}

} // namespace

namespace onert
{
namespace backend
{
namespace acl_common
{

template <typename T_ITensor, typename T_Tensor, typename T_SubTensor, typename T_MemoryManager,
          typename T_PoolManager, typename T_LifetimeManager, typename T_Allocator,
          typename T_MemoryGroup>
class AclLinearMemoryManager : public AclMemoryManager<T_ITensor, T_Tensor, T_SubTensor>
{
public:
  AclLinearMemoryManager()
    : _allocator{nullptr},
      _io_manager{createMemoryManager<T_MemoryManager, T_PoolManager, T_LifetimeManager>()},
      _io_group{std::make_shared<T_MemoryGroup>(_io_manager)}
  {
    // DO NOTHING
  }

  virtual ~AclLinearMemoryManager() = default;

  void allocate(void) override
  {
    _allocator = std::make_shared<T_Allocator>();
    _io_manager->populate(*_allocator, 1);
    _io_group->acquire();
  }

  void deallocate(void) override
  {
    _io_group->release();
    _io_manager->clear();
  }

  void startLifetime(const ir::OperandIndex &ind) override
  {
    auto &tensors = this->tensors();
    assert(tensors.find(ind) != tensors.end());

    auto tensor = tensors[ind];
    assert(tensor->handle());

    _io_group->manage(tensor->handle());
  }

  void finishLifetime(const ir::OperandIndex &ind) override
  {
    auto &tensors = this->tensors();
    assert(tensors.find(ind) != tensors.end());

    auto tensor = tensors[ind];
    assert(tensor->allocator());

    tensor->allocator()->allocate();
  }

private:
  std::shared_ptr<T_Allocator> _allocator;
  std::shared_ptr<T_MemoryManager> _io_manager;
  std::shared_ptr<T_MemoryGroup> _io_group;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_LINEAR_MEMORY_MANAGER_H__
