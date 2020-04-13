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
 * @file    View.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines internal::arm_compute::tensor::View class
 */
#ifndef __INTERNAL_ARM_COMPUTE_TENSOR_VIEW_H__
#define __INTERNAL_ARM_COMPUTE_TENSOR_VIEW_H__

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"

#include <arm_compute/core/ITensor.h>

namespace internal
{
namespace arm_compute
{
namespace tensor
{

/**
 * @brief Class to access tensor's element
 */
template <typename T> class View
{
public:
  /**
   * @brief     Construct a new View object
   * @param[in] tensor  Tensor to support access
   */
  View(::arm_compute::ITensor *tensor) : _tensor{tensor}
  {
    // DO NOTHING
  }

private:
  /**
   * @brief     Get offset of element in tensor
   * @param[in] index Index of element
   * @return    Offset of element
   */
  uint32_t byte_offset_of(const nnfw::misc::tensor::Index &index) const
  {
    // NOTE index.rank() >= _tensor->info()->num_dimensions() should hold here
    const uint32_t rank = index.rank();

    ::arm_compute::Coordinates coordinates;

    coordinates.set_num_dimensions(rank);

    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      coordinates[axis] = index.at(axis);
    }

    return _tensor->info()->offset_element_in_bytes(coordinates);
  }

public:
  /**
   * @brief     Get value of element in tensor
   * @param[in] index Index of element
   * @return    Value of element
   */
  T at(const nnfw::misc::tensor::Index &index) const
  {
    const auto offset = byte_offset_of(index);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

  /**
   * @brief     Get reference of element in tensor
   * @param[in] index Index of element
   * @return    Reference of element
   */
  T &at(const nnfw::misc::tensor::Index &index)
  {
    const auto offset = byte_offset_of(index);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

private:
  ::arm_compute::ITensor *_tensor;
};

} // namespace tensor
} // namespace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_TENSOR_VIEW_H__
