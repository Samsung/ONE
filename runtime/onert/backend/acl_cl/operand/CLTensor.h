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

#ifndef __ONERT_BACKEND_ACL_CL_OPERAND_CL_TENSOR_H__
#define __ONERT_BACKEND_ACL_CL_OPERAND_CL_TENSOR_H__

#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "ICLTensor.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{
namespace operand
{

class CLTensor : public ICLTensor
{
public:
  CLTensor() = delete;

public:
  CLTensor(const arm_compute::TensorInfo &info, size_t rank, size_t num_uses);

public:
  const arm_compute::CLTensor *handle() const override;
  arm_compute::CLTensor *handle() override;
  size_t num_uses() const { return _num_uses; }

public:
  arm_compute::CLTensorAllocator *allocator();
  /** Set given buffer as the buffer of the tensor
   *
   * @note Ownership of the memory is not transferred to this object.
   *       Thus management (allocate/free) should be done by the client.
   *
   * @param[in] host_ptr Storage to be used.
   */
  void setBuffer(void *host_ptr);

private:
  std::shared_ptr<arm_compute::CLTensor> _cl_tensor;
  size_t _num_uses;
};

} // namespace operand
} // namespace acl_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_CL_OPERAND_CL_TENSOR_H__
