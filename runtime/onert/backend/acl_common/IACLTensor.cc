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

#include "IACLTensor.h"
#include "Convert.h"
#include "Swizzle.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

size_t IACLTensor::calcOffset(const ir::Coordinates &coords) const
{
  auto rank = _rank;
  rank = rank == 0 ? 1 : rank;
  assert(static_cast<size_t>(rank) == coords.size());

  ::arm_compute::Coordinates acl_coords;
  for (size_t i = 0; i < rank; ++i)
  {
    const ARMComputeAxis reversed{static_cast<uint32_t>((rank - i) - 1)};
    acl_coords.set(reversed.value(), coords[i]);
  }

  return info()->offset_element_in_bytes(acl_coords);
}

ir::Layout IACLTensor::layout() const { return acl_common::asRuntimeLayout(info()->data_layout()); }

ir::DataType IACLTensor::data_type() const
{
  return acl_common::asRuntimeDataType(info()->data_type());
}

float IACLTensor::data_scale() const
{
  // FIXME What if quantization info is non-uniform?
  return info()->quantization_info().uniform().scale;
}

int32_t IACLTensor::data_zero_point() const
{
  // FIXME What if quantization info is non-uniform?
  return info()->quantization_info().uniform().offset;
}

} // namespace acl_common
} // namespace backend
} // namespace onert
