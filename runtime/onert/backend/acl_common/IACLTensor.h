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

#ifndef __ONERT_BACKEND_ACL_COMMON_I_ACL_TENSOR_H__
#define __ONERT_BACKEND_ACL_COMMON_I_ACL_TENSOR_H__

#include <backend/ITensor.h>
#include <arm_compute/core/ITensor.h>
#include "Swizzle.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

/**
 * @brief Class representing Tensor for ACL
 * @todo Override is_dynamic() method. We don't support dynamic tensor for ACL yet as of Apr, 2020.
 *       FYI, ACL ITensorInfo has is_dynamic() method, which seems currently not used.
 *       Maybe for ACL, this method can be implemented using ITensorInfo::is_dynamic() in future.
 */
class IACLTensor : public ITensor
{
public:
  IACLTensor() = default;
  IACLTensor(const IACLTensor &) = delete;
  IACLTensor &operator=(const IACLTensor &) = delete;
  IACLTensor(IACLTensor &&) = default;
  IACLTensor &operator=(IACLTensor &&) = default;

  IACLTensor(size_t rank) : _rank{rank} {}

public:
  uint8_t *buffer() const final { return handle()->buffer(); }
  size_t total_size() const final { return info()->total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const final;
  ir::Layout layout() const final;
  ir::DataType data_type() const final;
  float data_scale() const override;
  int32_t data_offset() const override;
  bool has_padding() const override { return info()->has_padding(); }
  bool is_dynamic() const override { return false; }
  ir::Shape getShape() const override
  {
    onert::ir::Shape shape(num_dimensions());
    for (uint32_t d = 0; d < num_dimensions(); d++)
      shape.dim(d) = dimension(d);
    return shape;
  }

public:
  virtual const arm_compute::ITensor *handle() const = 0;
  virtual arm_compute::ITensor *handle() = 0;

  const arm_compute::ITensorInfo *info() const { return handle()->info(); }
  arm_compute::ITensorInfo *info() { return handle()->info(); }

  size_t dimension(size_t index) const
  {
    // Assume that the front is higher dimensional.
    // i.g. N: 0, C: 1, H: 2, W: 3 for NCHW layout
    // NOTE This tensor must not be applied dim correction
    auto rank = _rank;
    rank = rank == 0 ? 1 : rank;
    assert(rank > index);
    const ARMComputeAxis reversed{(static_cast<uint32_t>(rank - index) - 1)};
    return info()->dimension(reversed.value());
  }
  size_t num_dimensions() const { return _rank; }

protected:
  size_t _rank; // Actual rank (reflects extended rank)
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif //__ONERT_BACKEND_ACL_COMMON_I_ACL_TENSOR_H__
