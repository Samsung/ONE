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
 * @brief   This file defines internal::arm_compute::matrix::View class
 */
#ifndef __INTERNAL_ARM_COMPUTE_MATRIX_VIEW_H__
#define __INTERNAL_ARM_COMPUTE_MATRIX_VIEW_H__

#include "misc/matrix/Shape.h"
#include "misc/matrix/Reader.h"

#include <arm_compute/core/ITensor.h>

namespace internal
{
namespace arm_compute
{
namespace matrix
{

/**
 * @brief Class to access matrix's element
 */
template <typename T> class View final : public nnfw::misc::matrix::Reader<T>
{
public:
  /**
   * @brief     Construct a new View object
   * @param[in] tensor  Matrix to support access
   */
  View(::arm_compute::ITensor *tensor) : _tensor{tensor}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief     Get value of element in matrix
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Value of element
   */
  T at(uint32_t row, uint32_t col) const override
  {
    const auto offset = matrix_index_to_byte_offset(row, col);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

public:
  /**
   * @brief     Get reference of element in matrix
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Refence of element
   */
  T &at(uint32_t row, uint32_t col)
  {
    const auto offset = matrix_index_to_byte_offset(row, col);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

private:
  /**
   * @brief     Get offset of element in matrix
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Offset of element
   */
  size_t matrix_index_to_byte_offset(uint32_t row, uint32_t col) const
  {
    return _tensor->info()->offset_element_in_bytes(::arm_compute::Coordinates{col, row});
  }

private:
  ::arm_compute::ITensor *_tensor;
};

} // namespace matrix
} // namespace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_MATRIX_VIEW_H__
