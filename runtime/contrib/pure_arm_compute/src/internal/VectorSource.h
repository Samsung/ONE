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
 * @file        VectorSource.h
 * @brief       This file contains VectorSource class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __INTERNAL_VECTOR_SOURCE_H__
#define __INTERNAL_VECTOR_SOURCE_H__

#include "internal/Source.h"

/**
 * @brief Class to store vector(2D) input data.
 * This is for push out the data to another tensor.
 * @tparam T Type of the data elements
 */
template <typename T> class VectorSource final : public Source
{
public:
  /**
   * @brief Construct a VectorSource object
   * @param[in] vlen Length of the vector
   * @param[in] base Base pointer of the actual data
   * @param[in] size Size of the data
   */
  VectorSource(const int32_t vlen, const T *base, const size_t size) : _vlen{vlen}, _base{base}
  {
    assert(size >= _vlen * sizeof(T));
  }

public:
  /**
   * @brief Push the data out to the another tensor
   * @param[out] The tensor that output data will be stored
   * @return N/A
   */
  void push(::arm_compute::ITensor &tensor) const override
  {
    for (int32_t n = 0; n < _vlen; ++n)
    {
      auto from = _base + n;
      auto into = reinterpret_cast<T *>(tensor.ptr_to_element(::arm_compute::Coordinates{n}));

      *into = *from;
    }
  }

private:
  const int32_t _vlen;
  const T *const _base;
};

#endif // __INTERNAL_VECTOR_SOURCE_H__
