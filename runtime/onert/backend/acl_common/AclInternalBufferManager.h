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

#ifndef __ONERT_BACKEND_ACL_COMMON_INTERNAL_BUFFER_MANAGER_H__
#define __ONERT_BACKEND_ACL_COMMON_INTERNAL_BUFFER_MANAGER_H__

#include <arm_compute/runtime/IMemoryManager.h>
#include <cassert>
#include <memory>

namespace onert
{
namespace backend
{
namespace acl_common
{

// NOTE. If any backend can use something like InternalBufferManager,
// this interface can be moved to core/include/backend/
/**
 * @brief Interface for InternalBufferManager which has ::arm_compute::IMemoryManager pointer
 */
struct IInternalBufferManager
{
  virtual ~IInternalBufferManager() = default;

  virtual void allocate(void) = 0;
  virtual void deallocate(void) = 0;

  /**
   * @brief Get shared_ptr of ::arm_compute::IMemoryManager
   */
  virtual std::shared_ptr<::arm_compute::IMemoryManager> internal_buffer_manager(void) = 0;
};

/**
 * @brief class for InternalBufferManager which has ::arm_compute::IMemoryManager pointer
 */
template <typename T_MemoryManager, typename T_PoolManager, typename T_LifetimeManager,
          typename T_Allocator>
class AclInternalBufferManager : public IInternalBufferManager
{
public:
  AclInternalBufferManager() : _allocator{nullptr}
  {
    std::shared_ptr<T_LifetimeManager> lifetime_mgr = std::make_shared<T_LifetimeManager>();
    std::shared_ptr<T_PoolManager> pool_mgr = std::make_shared<T_PoolManager>();

    _internal_manager = std::make_shared<T_MemoryManager>(lifetime_mgr, pool_mgr);
    assert(_internal_manager);
  }

  virtual ~AclInternalBufferManager() = default;

  /**
   * @brief Allocate the internal buffer manager on acl
   */
  void allocate(void) override
  {
    _allocator = std::make_shared<T_Allocator>();
    _internal_manager->populate(*_allocator, 1);
  }

  /**
   * @brief Deallocate the internal buffer manager on acl
   */
  void deallocate(void) override { _internal_manager->clear(); }

  /**
   * @brief Get shared_ptr of ::arm_compute::IMemoryManager
   */
  std::shared_ptr<::arm_compute::IMemoryManager> internal_buffer_manager(void) override
  {
    return _internal_manager;
  }

private:
  std::shared_ptr<T_Allocator> _allocator;
  std::shared_ptr<T_MemoryManager> _internal_manager;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_INTERNAL_BUFFER_MANAGER_H__
