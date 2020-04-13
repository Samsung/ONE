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
 * @file NonIncreasingStride.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains nnfw::misc::tensor::NonIncreasingStride class
 */
#ifndef __NNFW_MISC_TENSOR_NON_INCREASING_STRIDE_H__
#define __NNFW_MISC_TENSOR_NON_INCREASING_STRIDE_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"

#include <vector>

namespace nnfw
{
namespace misc
{
namespace tensor
{

/**
 * @brief Class to represent strides where stride[N-1] >= stride[N] holds for all N < rank
 */
class NonIncreasingStride
{
public:
  /**
   * @brief Initialize the stride data using @c Shape
   * @param[in] shape to build stride info
   * @return N/A
   */
  void init(const Shape &shape)
  {
    _stride.resize(shape.rank());

    // Scalar
    if (shape.rank() == 0)
      return;

    _stride.at(shape.rank() - 1) = 1;

    for (uint32_t axis = shape.rank() - 1; axis > 0; --axis)
    {
      _stride.at(axis - 1) = _stride.at(axis) * shape.dim(axis);
    }
  }

public:
  /**
   * @brief Get an stride value for specific axis
   * @param[in] axis   Axis of stride
   * @return The value of stride
   */
  uint32_t at(uint32_t axis) const { return _stride.at(axis); }

public:
  /**
   * @brief Get the 1-D offset of specified index for n-D tensor
   * @param index @c Index object
   * @return  1-D offset of index
   */
  uint32_t offset(const Index &index) const;

private:
  std::vector<uint32_t> _stride;
};

} // namespace tensor
} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_TENSOR_NON_INCREASING_STRIDE_H__
