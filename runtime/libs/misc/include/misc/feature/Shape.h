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
 * @file     Shape.h
 * @brief    This file contains Shape class for feature
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_FEATURE_SHAPE_H__
#define __NNFW_MISC_FEATURE_SHAPE_H__

#include <cstdint>

namespace nnfw
{
namespace misc
{
namespace feature
{

/**
 * @brief  Structure to have values of dimensions for feature
 */
struct Shape
{
  int32_t N; /**< The batch value  */
  int32_t C; /**< The depth value  */
  int32_t H; /**< The height value */
  int32_t W; /**< The width value  */

  /**
   * @brief  Construct Shape object using default constrcutor
   */
  Shape() = default;
  /**
   * @brief  Construct Shape object with three values of dimensions
   * @param[in]  depth  The depth value
   * @param[in]  height The height value
   * @param[in]  width  The width value
   */
  Shape(int32_t depth, int32_t height, int32_t width) : N{1}, C{depth}, H{height}, W{width}
  {
    // DO NOTHING
  }
  /**
   * @brief  Construct Shape object with four values of dimensions
   * @param[in]  batch  The batch value
   * @param[in]  depth  The depth value
   * @param[in]  height The height value
   * @param[in]  width  The width value
   */
  Shape(int32_t batch, int32_t depth, int32_t height, int32_t width)
    : N{batch}, C{depth}, H{height}, W{width}
  {
    // DO NOTHING
  }
};

} // namespace feature
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_FEATURE_H__
