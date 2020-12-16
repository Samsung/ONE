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

#ifndef __ONERT_BACKEND_ACL_NEON_OPERAND_NE_SUB_TENSOR_H__
#define __ONERT_BACKEND_ACL_NEON_OPERAND_NE_SUB_TENSOR_H__

#include <arm_compute/runtime/SubTensor.h>
#include "INETensor.h"

namespace onert
{
namespace backend
{
namespace acl_neon
{
namespace operand
{

class NESubTensor : public INETensor
{
public:
  NESubTensor() = delete;

public:
  NESubTensor(INETensor *parent, const arm_compute::TensorShape &tensor_shape,
              const arm_compute::Coordinates &coords, size_t rank, bool extend_parent = false);

public:
  const arm_compute::SubTensor *handle() const override;
  arm_compute::SubTensor *handle() override;

public:
  // This method is used to prevent the use of memcpy for SubTensor
  bool has_padding() const override { return true; }
  bool is_subtensor() const final { return true; }

private:
  std::shared_ptr<arm_compute::SubTensor> _ne_sub_tensor;
};

} // namespace operand
} // namespace acl_neon
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_NEON_OPERAND_NE_SUB_TENSOR_H__
