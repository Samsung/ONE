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
 * @brief   This file defines internal::nnapi::feature::Reader
 */
#ifndef __INTERNAL_NNAPI_FEATURE_READER_H__
#define __INTERNAL_NNAPI_FEATURE_READER_H__

#include "internal/nnapi/feature/Utils.h"

#include "misc/feature/Reader.h"

namespace internal
{
namespace nnapi
{
namespace feature
{

/**
 * @brief Class to support reading element in feature(3D, 4D)
 */
template <typename T> class Reader final : public nnfw::misc::feature::Reader<T>
{
public:
  /**
   * @brief     Construct a new Reader object
   * @param[in] shape Shape of feature
   * @param[in] ptr   Pointer to feature data
   * @param[in] len   Size of tensor (byte)
   */
  // NOTE The parameter len denotes the number of bytes.
  Reader(const ::nnfw::misc::feature::Shape &shape, const T *ptr, size_t len)
      : _shape{shape}, _ptr{ptr}
  {
    assert(shape.N * shape.C * shape.H * shape.W * sizeof(T) == len);
  }

public:
  /**
   * @brief   Get shape of feature
   * @return  Shape of feature
   */
  const nnfw::misc::feature::Shape &shape(void) const { return _shape; }

public:
  /**
   * @brief     Get value of element using channel, row, and column index for 3D feature
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Value of element
   */
  T at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    uint32_t index = index_of(_shape, ch, row, col);

    const auto arr = reinterpret_cast<const T *>(_ptr);

    return arr[index];
  }

  /**
   * @brief     Get value of element using batch, channel, row, and column index for 4D feature
   * @param[in] batch Batch index
   * @param[in] ch    Channel index
   * @param[in] row   Row index
   * @param[in] col   Column index
   * @return    Value of element
   */
  T at(uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    uint32_t index = index_of(_shape, batch, ch, row, col);

    return _ptr[index];
  }

private:
  nnfw::misc::feature::Shape _shape;

private:
  const T *_ptr;
};

} // namespace feature
} // namespace nnapi
} // namespace internal

#endif // __INTERNAL_NNAPI_FEATURE_READER_H__
