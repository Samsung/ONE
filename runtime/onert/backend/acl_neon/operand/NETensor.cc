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

#include <arm_compute/runtime/Memory.h>
#include <arm_compute/runtime/MemoryRegion.h>
#include "NETensor.h"

namespace onert
{
namespace backend
{
namespace acl_neon
{
namespace operand
{

NETensor::NETensor(const arm_compute::TensorInfo &info, size_t rank, size_t num_uses)
  : INETensor{rank}, _ne_tensor(std::make_shared<arm_compute::Tensor>()), _num_uses{num_uses}
{
  allocator()->init(info);
}

const arm_compute::Tensor *NETensor::handle() const { return _ne_tensor.get(); }

arm_compute::Tensor *NETensor::handle() { return _ne_tensor.get(); }

arm_compute::TensorAllocator *NETensor::allocator() { return _ne_tensor->allocator(); }

} // namespace operand
} // namespace acl_neon
} // namespace backend
} // namespace onert
