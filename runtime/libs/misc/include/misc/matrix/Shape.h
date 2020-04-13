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
 * @brief    This file contains Shape class for matrix
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_MISC_MATRIX_SHAPE_H__
#define __NNFW_MISC_MATRIX_SHAPE_H__

#include <cstdint>

namespace nnfw
{
namespace misc
{
namespace matrix
{

/**
 * @brief  Structure to have values of dimensions for matrix
 */
struct Shape
{
  int32_t H; /**< The height value */
  int32_t W; /**< The width value  */

  /**
   * @brief  Construct Shape object using default constrcutor
   */
  Shape() = default;

  /**
   * @brief  Construct Shape object with two values of dimensions
   * @param[in]  height The height value
   * @param[in]  width  The width value
   */
  Shape(int32_t height, int32_t width) : H{height}, W{width}
  {
    // DO NOTHING
  }
};

} // namespace matrix
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_MATRIX_SHAPE_H__
