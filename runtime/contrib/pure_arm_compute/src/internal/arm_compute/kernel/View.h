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
 * @brief   This file defines internel::arm_compute::kernel::View class
 */
#ifndef __INTERNAL_ARM_COMPUTE_KERNEL_VIEW_H__
#define __INTERNAL_ARM_COMPUTE_KERNEL_VIEW_H__

#include "misc/kernel/Shape.h"
#include "misc/kernel/Reader.h"

#include <arm_compute/core/ITensor.h>

namespace internal
{
namespace arm_compute
{
namespace kernel
{

/**
 * @brief Class to access kernel's element
 */
template <typename T> class View final : public nnfw::misc::kernel::Reader<T>
{
public:
  /**
   * @brief     Construct a new View object
   * @param[in] tensor  Kernel to support access
   */
  View(::arm_compute::ITensor *tensor) : _tensor{tensor}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief     Get value of element in kernel
   * @param[in] nth Kernel index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Value of element
   */
  T at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    const auto offset = kernel_index_to_byte_offset(nth, ch, row, col);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

public:
  /**
   * @brief     Get reference of element in kernel
   * @param[in] nth Kernel index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Reference of element
   */
  T &at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col)
  {
    const auto offset = kernel_index_to_byte_offset(nth, ch, row, col);

    T *ptr = reinterpret_cast<T *>(_tensor->buffer() + offset);

    return *ptr;
  }

private:
  /**
   * @brief     Get offset of element in kernel
   * @param[in] nth Kernel index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Offset of element
   */
  size_t kernel_index_to_byte_offset(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const
  {
    return _tensor->info()->offset_element_in_bytes(::arm_compute::Coordinates{col, row, ch, nth});
  }

private:
  ::arm_compute::ITensor *_tensor;
};

} // namespace kernel
} // namespace arm_compute
} // namespace internal

#endif // __INTERNAL_ARM_COMPUTE_FEATURE_VIEW_H__
