/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_CORE_RUNTIME_SHAPE_H
#define ONERT_MICRO_CORE_RUNTIME_SHAPE_H

#include "core/reader/OMCircleReader.h"

#include <cstdint>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace onert_micro::core
{

class OMRuntimeShape
{
  static const size_t kMaxDimsCount = 6;

private:
  size_t _size = 0;
  std::array<int32_t, kMaxDimsCount> _dims = {0};
  bool _is_scalar = false;

public:
  OMRuntimeShape() = default;

  // clang-format off

  OMRuntimeShape(const OMRuntimeShape &other)
    : _size(other._size)
    , _dims(other._dims)
  {}

  explicit OMRuntimeShape(size_t dimensions_count)
  {
    resize(dimensions_count);
  }

  template <size_t DimsCount>
  explicit OMRuntimeShape(const std::array<int32_t, DimsCount> &source_dims)
  {
    resize(DimsCount);
    std::copy(source_dims.cbegin(), source_dims.cend(), _dims.begin());
  }

  // clang-format on

  OMRuntimeShape(const circle::Tensor *tensor)
  {
    if (tensor == nullptr)
      return;

    auto shape = tensor->shape();

    if (shape == nullptr || shape->size() == 0)
    {
      _is_scalar = true;
      _size = 1;
      _dims[0] = 1;

      return;
    }

    _size = shape->size();
    std::copy(shape->cbegin(), shape->cend(), _dims.begin());
  }

  OMRuntimeShape(size_t new_shape_size, const OMRuntimeShape &shape, int32_t pad_value)
  {
    assert(new_shape_size >= shape._size);
    resize(new_shape_size);

    const size_t size_increase = new_shape_size - shape._size;

    for (auto i = 0u; i < size_increase; ++i)
    {
      setDim(i, pad_value);
    }

    auto from = shape._dims.cbegin();
    auto to = _dims.begin() + size_increase;

    std::copy(from, from + shape._size, to);
  }

  OMRuntimeShape(size_t shape_size, int32_t value)
  {
    resize(shape_size);

    for (auto i = 0u; i < shape_size; ++i)
    {
      setDim(i, value);
    }
  }

  static OMRuntimeShape extendedShape(size_t new_shape_size, const OMRuntimeShape &shape)
  {
    return OMRuntimeShape(new_shape_size, shape, 1);
  }

  bool operator==(const OMRuntimeShape &other) const
  {
    return _size == other._size && _dims == other._dims;
  }

  size_t flatSize() const
  {
    if (_size == 0)
      return 0;

    auto it = _dims.cbegin();

    return std::accumulate(it, it + _size, 1u, std::multiplies<size_t>());
  }

  // clang-format off

  bool isScalar() const
  {
    return _is_scalar;
  }

  int32_t *dimsData()
  {
    return _dims.data();
  }

  const int32_t *dimsData() const
  {
    return _dims.data();
  }

  size_t dimensionsCount() const
  {
    return _size;
  }

  int32_t dims(size_t i) const
  {
    assert(i <= _size);
    return _dims[i];
  }

  void setDim(size_t i, int32_t val)
  {
    assert(i <= _size);
    _dims[i] = val;
  }

  void resize(size_t dimensions_count)
  {
    assert(dimensions_count <= kMaxDimsCount);
    _size = dimensions_count;
  }
};

} // namespace onert_micro::core

#endif // ONERT_MICRO_CORE_RUNTIME_SHAPE_H
