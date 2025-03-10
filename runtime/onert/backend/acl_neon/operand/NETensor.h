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

#ifndef __ONERT_BACKEND_ACL_NEON_OPERAND_NE_TENSOR_H__
#define __ONERT_BACKEND_ACL_NEON_OPERAND_NE_TENSOR_H__

#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/runtime/Tensor.h>
#include "arm_compute/runtime/TensorAllocator.h"
#include "INETensor.h"

namespace onert::backend::acl_neon::operand
{

class NETensor : public INETensor
{
public:
  NETensor() = delete;

public:
  NETensor(const arm_compute::TensorInfo &info, size_t rank, size_t num_uses);

public:
  const arm_compute::Tensor *handle() const override;
  arm_compute::Tensor *handle() override;
  size_t num_uses() const { return _num_uses; }

public:
  arm_compute::TensorAllocator *allocator();

private:
  std::shared_ptr<arm_compute::Tensor> _ne_tensor;
  size_t _num_uses;
};

} // namespace onert::backend::acl_neon::operand

#endif // __ONERT_BACKEND_ACL_NEON_OPERAND_NE_TENSOR_H__
