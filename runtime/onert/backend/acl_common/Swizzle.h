/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_ACL_COMMON_SWIZZLE_H__
#define __ONERT_BACKEND_ACL_COMMON_SWIZZLE_H__

#include <cassert>
#include <ir/Layout.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

class ARMComputeAxis
{
public:
  ARMComputeAxis() = default;

public:
  explicit ARMComputeAxis(uint32_t value) : _value{value}
  {
    // DO NOTHING
  }

public:
  uint32_t value(void) const { return _value; }

private:
  uint32_t _value;
};

// Convert axis in acl order
inline ARMComputeAxis ToARMComputeAxis(uint32_t rank, uint32_t axis)
{
  assert(rank > axis);

  return ARMComputeAxis{(rank - axis) - 1};
}

inline ::arm_compute::Coordinates getARMComputeAxises(uint32_t rank)
{
  ::arm_compute::Coordinates res{};

  res.set_num_dimensions(rank);

  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    res.set(axis, ToARMComputeAxis(rank, axis).value());
  }

  return res;
}

// Restructure runtime_permutationVector to ACL_permutationVector
inline ::arm_compute::PermutationVector
getARMComputePermutationVector(uint32_t rank, const std::vector<int32_t> runtime_pv)
{
  // rank upto 4 is supported
  assert(rank <= 4);
  assert(runtime_pv.size() > 0);

  int new_pv[4] = {0};
  ::arm_compute::Coordinates axises = getARMComputeAxises(rank);

  for (uint32_t i = 0; i < rank; ++i)
  {
    new_pv[axises[i]] = ToARMComputeAxis(rank, runtime_pv[i]).value();
  }

  ::arm_compute::PermutationVector ACL_PV =
    ::arm_compute::PermutationVector{new_pv[0], new_pv[1], new_pv[2], new_pv[3]};
  ACL_PV.set_num_dimensions(rank);

  return ACL_PV;
}

template <typename T> inline T ReorderBits(T in, size_t numOfBits)
{
  assert(numOfBits > 0);
  T out = 0;
  for (int32_t i = numOfBits - 1; i >= 0; --i)
  {
    const uint32_t toShift = numOfBits - ToARMComputeAxis(numOfBits, i).value() - 1;
    out += ((in & 1) << toShift);
    in >>= 1;
  }
  return out;
}

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_SWIZZLE_H__
