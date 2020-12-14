/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CLTensor.h"

#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLMemory.h>
#include <arm_compute/runtime/CL/CLMemoryRegion.h>

#include <Convert.h>

namespace onert
{
namespace backend
{
namespace acl_cl
{
namespace operand
{

CLTensor::CLTensor(const arm_compute::TensorInfo &info, size_t rank, size_t num_uses)
  : ICLTensor{rank}, _cl_tensor(std::make_shared<arm_compute::CLTensor>()), _num_uses{num_uses}
{
  allocator()->init(info);
}

const arm_compute::CLTensor *CLTensor::handle() const { return _cl_tensor.get(); }

arm_compute::CLTensor *CLTensor::handle() { return _cl_tensor.get(); }

arm_compute::CLTensorAllocator *CLTensor::allocator() { return _cl_tensor->allocator(); }

void CLTensor::setBuffer(void *host_ptr)
{
  // Constructs a Buffer on a user-supplied memory
  auto buffer = cl::Buffer(arm_compute::CLScheduler::get().context(),
                           CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, info()->total_size(), host_ptr);
  // import memory
  allocator()->import_memory(buffer);
}

} // namespace operand
} // namespace acl_cl
} // namespace backend
} // namespace onert
