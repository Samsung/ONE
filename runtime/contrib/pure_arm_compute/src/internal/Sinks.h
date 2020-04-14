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
 * @file        Sinks.h
 * @brief       This file contains TensorSink class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __INTERNAL_SINKS_H__
#define __INTERNAL_SINKS_H__

#include "internal/Sink.h"

// TODO Extract TensorSink into TensorSink.h
//
// TensorSink
//
#include "internal/Swizzle.h"

#include "internal/nnapi/tensor/View.h"
#include "internal/arm_compute/tensor/View.h"

#include "misc/tensor/IndexIterator.h"

/**
 * @brief Class to store NN model output data for general-shaped tensors.
 * This is for pulling data to internal tensor from other tensor.
 * @tparam T Type of the data elements
 */
template <typename T> class TensorSink final : public Sink
{
public:
  /**
   * @brief Construct a TensorSink object
   *
   * @param[in] shape general-shaped tensor dimensions
   * @param[in] base Base pointer of the actual data
   * @param[in] size Size of the data
   */
  TensorSink(const nnfw::misc::tensor::Shape &shape, T *base, const size_t size)
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
    const ::internal::arm_compute::tensor::View<T> from{&tensor};
    ::internal::nnapi::tensor::View<T> into{_shape, _base, _size};

    using ::nnfw::misc::tensor::Index;
    using ::nnfw::misc::tensor::iterate;

    const uint32_t rank = _shape.rank();

    ::nnfw::misc::tensor::iterate(_shape) << [&](const Index &raw) {
      Index permuted(raw.rank());

      for (uint32_t axis = 0; axis < rank; ++axis)
      {
        permuted.at(ToARMComputeAxis(rank, axis).value()) = raw.at(axis);
      }

      const auto value = from.at(permuted);
      into.at(raw) = value;
    };
  }

private:
  const nnfw::misc::tensor::Shape _shape;

private:
  T *const _base;
  const size_t _size;
};

#endif // __INTERNAL_SINKS_H__
