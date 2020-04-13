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
 * @file     fp32.h
 * @brief    This file contains functions to compare float values
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_FP32_H__
#define __NNFW_MISC_FP32_H__

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cstdint>

namespace nnfw
{
namespace misc
{
namespace fp32
{

/**
 * @brief   Get the difference between two float values as a relative value.
 * @param[in]   lhs   A float value to be compared
 * @param[in]   rhs   A float value to be compared
 * @return  A relative value of difference between two float values.
 */
inline float relative_diff(float lhs, float rhs)
{
  const auto diff = std::fabs(lhs - rhs);
  const auto base = std::max(std::fabs(lhs), std::fabs(rhs));

  return diff / base;
}

/**
 * @brief   Verify that an obtained float value is equal to the expected float value
 *          by using FLT_EPSILON
 * @param[in]   expected   An expected float value to be compared
 * @param[in]   obtained   An obtained float value to be compared
 * @param[in]   tolerance  A tolerance value
 * @return  @c true if both values are equal, otherwise @c false
 */
inline bool epsilon_equal(float expected, float obtained, uint32_t tolerance = 1)
{
  if (std::isnan(expected) && std::isnan(obtained))
  {
    return true;
  }

  // Let's use relative epsilon comparision
  const auto diff = std::fabs(expected - obtained);
  const auto max = std::max(std::fabs(expected), std::fabs(obtained));

  return diff <= (max * FLT_EPSILON * tolerance);
}

/**
 * @brief   Verify that an obtained float value is equal to the expected float value
 *          by comparing absolute tolerance value
 * @param[in]   expected   An expected float value to be compared
 * @param[in]   obtained   An obtained float value to be compared
 * @param[in]   tolerance  A tolerance value
 * @return  @c true if both values are equal, otherwise @c false
 */
inline bool absolute_epsilon_equal(float expected, float obtained, float tolerance = 0.001)
{
  if (std::isnan(expected) && std::isnan(obtained))
  {
    return true;
  }

  // Let's use absolute epsilon comparision
  const auto diff = std::fabs(expected - obtained);

  return diff <= tolerance;
}

} // namespace fp32
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_FP32_H__
