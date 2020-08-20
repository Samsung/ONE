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

#ifndef __ONERT_BACKEND_ACL_CL_TENSOR_MANAGER_H__
#define __ONERT_BACKEND_ACL_CL_TENSOR_MANAGER_H__

#include <arm_compute/runtime/CL/CLBufferAllocator.h>
#include <arm_compute/runtime/PoolManager.h>
#include <arm_compute/runtime/BlobLifetimeManager.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/MemoryGroup.h>

#include <AclMemoryManager.h>
#include <AclLinearMemoryManager.h>
#include <AclInternalBufferManager.h>
#include <AclTensorManager.h>

#include "operand/CLTensor.h"
#include "operand/CLSubTensor.h"

#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{

using MemoryManager =
    acl_common::AclMemoryManager<operand::ICLTensor, operand::CLTensor, operand::CLSubTensor>;

using LinearMemoryManager = acl_common::AclLinearMemoryManager<
    operand::ICLTensor, operand::CLTensor, operand::CLSubTensor,
    ::arm_compute::MemoryManagerOnDemand, ::arm_compute::PoolManager,
    ::arm_compute::BlobLifetimeManager, ::arm_compute::CLBufferAllocator,
    ::arm_compute::MemoryGroup>;

using InternalBufferManager = acl_common::AclInternalBufferManager<
    ::arm_compute::MemoryManagerOnDemand, ::arm_compute::PoolManager,
    ::arm_compute::BlobLifetimeManager, ::arm_compute::CLBufferAllocator>;

using TensorManager =
    acl_common::AclTensorManager<operand::ICLTensor, operand::CLTensor, operand::CLSubTensor>;

inline TensorManager *createTensorManager(bool is_linear_executor)
{
  if (is_linear_executor)
  {
    VERBOSE(acl_cl_createTensorManager) << "AclTensorManager as Linear" << std::endl;
    return new TensorManager(new MemoryManager(), new LinearMemoryManager(),
                             new InternalBufferManager());
  }
  else
  {
    VERBOSE(acl_cl_createTensorManager) << "AclTensorManager" << std::endl;
    return new TensorManager(new MemoryManager(), new MemoryManager(), new InternalBufferManager());
  }
}

} // namespace acl_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_CL_TENSOR_MANAGER_H__
