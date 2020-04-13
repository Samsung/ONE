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
 * @brief   This file defines internal::nnapi::feature::View class
 */
#ifndef __INTERNAL_NNAPI_FEATURE_VIEW_H__
#define __INTERNAL_NNAPI_FEATURE_VIEW_H__

#include "internal/nnapi/feature/Utils.h"

#include "misc/feature/Reader.h"

namespace internal
{
namespace nnapi
{
namespace feature
{

/**
 * @brief   Class to access feature's element information using index
 */
template <typename T> class View final : public nnfw::misc::feature::Reader<T>
{
public:
  /**
   * @brief     Construct a new View object
   * @param[in] shape Shape of feature
   * @param[in] ptr   Pointer to feature data
   * @param[in] len   Size of feature (byte)
   * @return
   */
  // NOTE The parameter len denotes the number of bytes.
  View(const ::nnfw::misc::feature::Shape &shape, T *ptr, size_t len) : _shape{shape}, _ptr{ptr}
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
   * @brief     Get value of element in 3D feature using channel, row, and column index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Value of element
   */
  T at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    uint32_t index = index_of(_shape, ch, row, col);

    return _ptr[index];
  }

  /**
   * @brief     Get value of element in 4D feature using batch, channel, row and column index
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

  /**
   * @brief     Get reference of element in 3D feature using channel, row, and column index
   * @param[in] ch  Channel index
   * @param[in] row Row index
   * @param[in] col Column index
   * @return    Reference of element
   */
  T &at(uint32_t ch, uint32_t row, uint32_t col)
  {
    uint32_t index = index_of(_shape, ch, row, col);

    return _ptr[index];
  }

  /**
   * @brief     Get reference of element in 4D feature using batch, channel, row and column index
   * @param[in] batch Batch index
   * @param[in] ch    Channel index
   * @param[in] row   Row index
   * @param[in] col   Column index
   * @return    Reference of element
   */
  T &at(uint32_t batch, uint32_t ch, uint32_t row, uint32_t col)
  {
    uint32_t index = index_of(_shape, batch, ch, row, col);

    return _ptr[index];
  }

private:
  nnfw::misc::feature::Shape _shape;

private:
  T *_ptr;
};

} // namespace feature
} // namespace nnapi
} // namespace internal

#endif // __INTERNAL_NNAPI_FEATURE_VIEW_H__
