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

#ifndef __ONERT_BACKEND_ACL_CL_OPERAND_I_CL_TENSOR_H__
#define __ONERT_BACKEND_ACL_CL_OPERAND_I_CL_TENSOR_H__

#include <arm_compute/core/CL/ICLTensor.h>

#include <IACLTensor.h>

namespace onert
{
namespace backend
{
namespace acl_cl
{
namespace operand
{

class ICLTensor : public acl_common::IACLTensor
{
public:
  ICLTensor(size_t rank) : IACLTensor{rank} {}
  const arm_compute::ICLTensor *handle() const override = 0;
  arm_compute::ICLTensor *handle() override = 0;

public:
  void access(const std::function<void(ITensor &tensor)> &fn) final;
  bool needMemoryMap() const final { return true; }
  void enqueueWriteBuffer(const void *ptr, bool blocking = true) final;
  void enqueueReadBuffer(void *ptr, bool blocking = true) final;

private:
  void map(cl::CommandQueue &q, bool blocking = true) { return handle()->map(q, blocking); }
  void unmap(cl::CommandQueue &q) { return handle()->unmap(q); }
};

} // namespace operand
} // namespace acl_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_CL_OPERAND_I_CL_TENSOR_H__
