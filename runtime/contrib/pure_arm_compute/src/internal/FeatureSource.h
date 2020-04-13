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
 * @file        FeatureSource.h
 * @brief       This file contains FeatureSource class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __INTERNAL_FEATURE_SOURCE_H__
#define __INTERNAL_FEATURE_SOURCE_H__

#include <misc/feature/Shape.h>
#include <misc/feature/IndexIterator.h>

#include "internal/nnapi/feature/Reader.h"
#include "internal/arm_compute/feature/View.h"

/**
 * @brief Class to store feature(4D) input data.
 * This is for push out the data to another tensor.
 * @tparam T Type of the data elements
 */
template <typename T> class FeatureSource final : public Source
{
public:
  /**
   * @brief Construct a FeatureSource object
   *
   * @param[in] shape 4D tensor dimensions for this feature
   * @param[in] base Base pointer of the actual data
   * @param[in] size Size of the data
   */
  FeatureSource(const nnfw::misc::feature::Shape &shape, const T *base, const size_t size)
      : _shape{shape}, _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Push the data out to the another tensor
   * @param[out] The tensor that output data will be stored
   * @return N/A
   */
  void push(::arm_compute::ITensor &tensor) const override
  {
    const ::internal::nnapi::feature::Reader<T> from{_shape, _base, _size};
    ::internal::arm_compute::feature::View<T> into{&tensor};

    ::nnfw::misc::feature::iterate(_shape)
        << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
             const auto value = from.at(batch, ch, row, col);
             into.at(batch, ch, row, col) = value;
           };
  }

private:
  const nnfw::misc::feature::Shape _shape;
  const T *const _base;
  const size_t _size;
};

#endif // __INTERNAL_FEATURE_SOURCE_H__
