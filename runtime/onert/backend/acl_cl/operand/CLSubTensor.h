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

#ifndef __ONERT_BACKEND_ACL_CL_OPERAND_CL_SUB_TENSOR_H__
#define __ONERT_BACKEND_ACL_CL_OPERAND_CL_SUB_TENSOR_H__

#include <arm_compute/runtime/CL/CLSubTensor.h>
#include "ICLTensor.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{
namespace operand
{

class CLSubTensor : public ICLTensor
{
public:
  CLSubTensor() = delete;

public:
  CLSubTensor(ICLTensor *parent, const arm_compute::TensorShape &tensor_shape,
              const arm_compute::Coordinates &coords, size_t rank, bool extend_parent = false);

public:
  size_t num_dimensions() const final { return _rank; }

public:
  const arm_compute::CLSubTensor *handle() const override;
  arm_compute::CLSubTensor *handle() override;

public:
  // This method is used to prevent the use of memcpy for SubTensor
  bool has_padding() const override { return true; }
  bool is_subtensor() const final { return true; }

private:
  std::shared_ptr<arm_compute::CLSubTensor> _cl_sub_tensor;
  size_t _rank;
};

} // namespace operand
} // namespace acl_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_CL_OPERAND_CL_SUB_TENSOR_H__
