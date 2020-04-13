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
 * @file        FeatureSink.h
 * @brief       This file contains FeatureSink class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __INTERNAL_FEATURE_SINK_H__
#define __INTERNAL_FEATURE_SINK_H__

#include "internal/Sink.h"
#include "internal/nnapi/feature/View.h"
#include "internal/arm_compute/feature/View.h"

#include <misc/feature/Shape.h>
#include "misc/feature/IndexIterator.h"

/**
 * @brief Class to store Feature(4D) output data.
 * This is for pulling data to internal tensor from other tensor.
 * @tparam T Type of the data elements
 */
template <typename T> class FeatureSink final : public Sink
{
public:
  /**
   * @brief Construct a FeatureSink object
   *
   * @param[in] shape 4D tensor dimensions for this feature
   * @param[in] base Base pointer of the actual data
   * @param[in] size Size of the data
   */
  FeatureSink(const nnfw::misc::feature::Shape &shape, T *base, const size_t size)
      : _shape{shape}, _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Pull the data into the internal structure
   * @param[in] tensor The tensor which contains source data
   * @return N/A
   */
  void pull(::arm_compute::ITensor &tensor) const override
  {
    const ::internal::arm_compute::feature::View<T> from{&tensor};
    // TODO Should remove casting.
    // Inevitably casting must be done.
    ::internal::nnapi::feature::View<T> into{_shape, _base, _size};

    ::nnfw::misc::feature::iterate(_shape)
        << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
             const auto value = from.at(batch, ch, row, col);
             into.at(batch, ch, row, col) = value;
           };
  }

private:
  const nnfw::misc::feature::Shape _shape;
  T *const _base;
  const size_t _size;
};

#endif // __INTERNAL_FEATURE_SINK_H__
