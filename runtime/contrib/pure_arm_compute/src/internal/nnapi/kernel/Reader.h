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
 * @file    Reader.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines internal::nnapi::kernel::Reader class
 */
#ifndef __INTERNAL_NNAPI_KERNEL_READER_H__
#define __INTERNAL_NNAPI_KERNEL_READER_H__

#include "misc/kernel/Shape.h"
#include "misc/kernel/Reader.h"

namespace internal
{
namespace nnapi
{
namespace kernel
{

/**
 * @brief Class to support reading element in kernel
 */
template <typename T> class Reader final : public nnfw::misc::kernel::Reader<T>
{
public:
  /**
   * @brief     Construct a new Reader object
   * @param[in] shape Shape of kernel
   * @param[in] ptr   Pointer to kernel data
   * @param[in] len   Size of kernel (byte)
   */
  // NOTE The parameter len denotes the number of bytes.
  Reader(const ::nnfw::misc::kernel::Shape &shape, const T *ptr, size_t len)
      : _shape{shape}, _ptr{ptr}
  {
    assert(shape.N * shape.C * shape.H * shape.W * sizeof(T) == len);
  }

public:
  /**
   * @brief   Get shape of kernel
   * @return  Shape of kernel
   */
  const nnfw::misc::kernel::Shape &shape(void) const { return _shape; }

public:
  /**
   * @brief     Get value of element for kernel
   * @param[in] nth Kernel index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Value of element
   */
  T at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    // NNAPI uses NHWC ordering
    uint32_t index = 0;

    index += nth * _shape.H * _shape.W * _shape.C;
    index += row * _shape.W * _shape.C;
    index += col * _shape.C;
    index += ch;

    return _ptr[index];
  }

private:
  nnfw::misc::kernel::Shape _shape;

private:
  const T *_ptr;
};

} // namespace kernel
} // namespace nnapi
} // namespace internal

#endif // __INTERNAL_NNAPI_KERNEL_READER_H__
