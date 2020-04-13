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
 * @file        VectorSink.h
 * @brief       This file contains VectorSink class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __INTERNAL_VECTOR_SINK_H__
#define __INTERNAL_VECTOR_SINK_H__

#include "internal/Sink.h"

#include <arm_compute/core/ITensor.h>

#include <cassert>

/**
 * @brief Class to store vector(2D) output data.
 * This is for pulling out the data to another tensor.
 * @tparam T Type of the data elements
 */
template <typename T> class VectorSink final : public Sink
{
public:
  /**
   * @brief Construct a VectorSink object
   * @param[in] vlen Length of the vector
   * @param[in] base Base pointer of the actual data
   * @param[in] size Size of the data
   */
  VectorSink(const int32_t vlen, T *base, const size_t size) : _vlen{vlen}, _base{base}
  {
    assert(size >= _vlen * sizeof(T));
  }

public:
  /**
   * @brief Pull the data into the internal structure
   * @param[in] tensor The tensor which contains source data
   * @return N/A
   */
  void pull(::arm_compute::ITensor &tensor) const override
  {
    for (int32_t n = 0; n < _vlen; ++n)
    {
      auto from = reinterpret_cast<T *>(tensor.ptr_to_element(::arm_compute::Coordinates{n}));
      auto into = _base + n;

      *into = *from;
    }
  }

private:
  const int32_t _vlen;
  T *const _base;
};

#endif // __INTERNAL_VECTOR_SINK_H__
