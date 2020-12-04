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
 * @brief    This file contains Shape structure
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_KERNEL_SHAPE_H__
#define __NNFW_MISC_KERNEL_SHAPE_H__

#include <cstdint>

namespace nnfw
{
namespace misc
{
namespace kernel
{

/**
 * @brief Structure to Shape
 */
struct Shape
{
  int32_t N; /**< The kernel index */
  int32_t C; /**< The channel index */
  int32_t H; /**< The height index */
  int32_t W; /**< The width index */

  /**
   * @brief Construct a new Shape object as default
   */
  Shape() = default;

  /**
   * @brief Construct a new Shape object with parameters
   * @param[in] count The kernel index
   * @param[in] depth The channel index
   * @param[in] height The height index
   * @param[in] width The width index
   */
  Shape(int32_t count, int32_t depth, int32_t height, int32_t width)
    : N{count}, C{depth}, H{height}, W{width}
  {
    // DO NOTHING
  }
};

} // namespace kernel
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_KERNEL_SHAPE_H__
