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

/**
 * @file    Swizzle.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines ARMComputeAxis class and utility functions to support mapping
 *          between arm compute axis and NNAPI axis
 */
#ifndef __SWIZZLE_H__
#define __SWIZZLE_H__

/**
 * @brief Class to represent arm compute axis
 */
class ARMComputeAxis
{
public:
  /**
   * @brief Construct a new ARMComputeAxis object
   */
  ARMComputeAxis() = default;

public:
  /**
   * @brief Construct a new ARMComputeAxis object
   * @param[in] value Raw axis number
   */
  explicit ARMComputeAxis(uint32_t value) : _value{value}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief   Get raw axis number
   * @return  Raw axis number
   */
  uint32_t value(void) const { return _value; }

private:
  uint32_t _value;
};

/**
 * @brief     Convert T/F Lite / NNAPI axis (based on ...NHWC) to arm compute axis (WHCN...)
 * @param[in] rank  Rank of shape
 * @param[in] axis  Axis to map
 * @return    ARMComputeAxis including arm compute axis info
 */
inline ARMComputeAxis ToARMComputeAxis(uint32_t rank, uint32_t axis)
{
  assert(rank > axis);
  const ARMComputeAxis reversed{(rank - axis) - 1};

  if (rank < 4)
  {
    return reversed;
  }

  // DEPTH
  if (0 == reversed.value())
  {
    return ARMComputeAxis{2};
  }
  // WIDTH
  if (1 == reversed.value())
  {
    return ARMComputeAxis{0};
  }
  // HEIGHT
  if (2 == reversed.value())
  {
    return ARMComputeAxis{1};
  }

  // ELSE
  return reversed;
}

#include <cassert>

/**
 * @brief     Covert bitmask info from NNAPI axis to arm compute axis
 * @param[in] in        Bitmask data
 * @param[in] numOfBits Used bits (rank)
 * @return    Coverted bitmask
 */
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

#endif // __SWIZZLE_H__
