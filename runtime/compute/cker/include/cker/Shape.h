/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_SHAPE_H__
#define __NNFW_CKER_SHAPE_H__

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iterator>
#include <variant>
#include <vector>

namespace nnfw
{
namespace cker
{

class Shape
{
public:
  // Shapes with dimensions up to 6 are stored directly in the structure, while
  // larger shapes are separately allocated.
  static constexpr int kMaxSmallSize = 6;

  // Delete copy assignment operator.
  Shape &operator=(Shape const &) = delete;

  // Default constructor: initializes an empty shape (size = 0) with small storage.
  Shape() : _size(0), dims_(std::array<int32_t, kMaxSmallSize>{}) {}

  // Constructor that takes a dimension count.
  // If dimensions_count <= kMaxSmallSize, it uses a fixed-size array.
  // Otherwise, it uses a dynamic vector.
  explicit Shape(int dimensions_count) : _size(dimensions_count) { initStorage(dimensions_count); }

  // Constructor that creates a shape of given size and fills all dimensions with "value".
  Shape(int shape_size, int32_t value) : _size(shape_size)
  {
    initStorage(shape_size);
    for (int i = 0; i < shape_size; ++i)
    {
      SetDim(i, value);
    }
  }

  // Constructor that creates a shape from an array of dimension data.
  Shape(int dimensions_count, const int32_t *dims_data) : _size(dimensions_count)
  {
    initStorage(dimensions_count);
    ReplaceWith(dimensions_count, dims_data);
  }

  // Initializer list constructor.
  // Marked explicit to avoid unintended overload resolution.
  Shape(const std::initializer_list<int> init_list) : _size(0)
  {
    const auto size = static_cast<int>(std::distance(init_list.begin(), init_list.end()));
    initStorage(size);
    BuildFrom(init_list);
  }

  // Copy constructor
  Shape(const Shape &other) : _size(other._size)
  {
    if (_size <= kMaxSmallSize)
    {
      // When the number of dimensions is small, copy the fixed array.
      dims_ = std::get<std::array<int32_t, kMaxSmallSize>>(other.dims_);
    }
    else
    {
      // Otherwise, copy the dynamically allocated vector.
      dims_ = std::get<std::vector<int32_t>>(other.dims_);
    }
  }
  Shape(Shape &&other) = default;

  bool operator==(const Shape &comp) const
  {
    return this->_size == comp._size &&
           std::memcmp(DimsData(), comp.DimsData(), _size * sizeof(int32_t)) == 0;
  }

  ~Shape() = default;

  // Returns the number of dimensions.
  inline int32_t DimensionsCount() const { return _size; }

  // Returns the dimension size at index i.
  inline int32_t Dims(int i) const
  {
    assert(i >= 0 && i < _size);
    if (_size <= kMaxSmallSize)
    {
      return std::get<std::array<int32_t, kMaxSmallSize>>(dims_)[i];
    }
    else
    {
      return std::get<std::vector<int32_t>>(dims_)[i];
    }
  }

  // Sets the dimension at index i.
  inline void SetDim(int i, int32_t val)
  {
    assert(i >= 0 && i < _size);
    if (_size <= kMaxSmallSize)
    {
      std::get<std::array<int32_t, kMaxSmallSize>>(dims_)[i] = val;
    }
    else
    {
      std::get<std::vector<int32_t>>(dims_)[i] = val;
    }
  }

  // Returns a pointer to the dimension data (mutable).
  inline int32_t *DimsData()
  {
    if (_size <= kMaxSmallSize)
    {
      return std::get<std::array<int32_t, kMaxSmallSize>>(dims_).data();
    }
    else
    {
      return std::get<std::vector<int32_t>>(dims_).data();
    }
  }

  // Returns a pointer to the dimension data (const).
  inline const int32_t *DimsData() const
  {
    if (_size <= kMaxSmallSize)
    {
      return std::get<std::array<int32_t, kMaxSmallSize>>(dims_).data();
    }
    else
    {
      return std::get<std::vector<int32_t>>(dims_).data();
    }
  }

  // The caller must ensure that the shape is no larger than 6D.
  inline const int32_t *DimsDataUpTo6D() const
  {
    return std::get<std::array<int32_t, kMaxSmallSize>>(dims_).data();
  }

  // Resizes the shape to dimensions_count while preserving existing data.
  inline void Resize(int dimensions_count)
  {
    // If dims_ is in a valueless state (i.e. not yet initialized or lost due to an exception),
    // initialize dims_ explicitly based on dimensions_count to ensure it is in a valid state.
    if (dims_.valueless_by_exception())
    {
      initStorage(dimensions_count);
    }

    std::vector<int32_t> oldDims;
    oldDims.reserve(_size);
    if (_size <= kMaxSmallSize)
    {
      const auto &arr = std::get<std::array<int32_t, kMaxSmallSize>>(dims_);
      oldDims.assign(arr.begin(), arr.begin() + _size);
    }
    else
    {
      oldDims = std::get<std::vector<int32_t>>(dims_);
    }

    int count = std::min(_size, dimensions_count);

    if (dimensions_count <= kMaxSmallSize)
    {
      std::array<int32_t, kMaxSmallSize> dims = {};
      std::copy_n(oldDims.begin(), count, dims.begin());
      dims_ = dims;
    }
    else
    {
      std::vector<int32_t> dims(dimensions_count, 0);
      std::copy_n(oldDims.begin(), count, dims.begin());
      dims_ = dims;
    }

    _size = dimensions_count;
  }

  // Replaces the current shape with a new one defined by dimensions_count and dims_data.
  inline void ReplaceWith(int dimensions_count, const int32_t *dims_data)
  {
    // Allow dims_data to be nullptr when dimensions_count is 0,
    // because there are no dimensions to copy. For any non-zero dimensions_count,
    // dims_data must not be nullptr to ensure valid shape data is provided.
    assert(dimensions_count == 0 || dims_data != nullptr);
    Resize(dimensions_count);
    std::memcpy(DimsData(), dims_data, dimensions_count * sizeof(int32_t));
  }

  // Replaces the current shape with another shape.
  inline void ReplaceWith(const Shape &other)
  {
    ReplaceWith(other.DimensionsCount(), other.DimsData());
  }

  // Replaces the current shape with another shape using move semantics.
  inline void ReplaceWith(Shape &&other)
  {
    std::swap(_size, other._size);
    dims_ = std::move(other.dims_);
  }

  // Builds the shape from an iterable sequence.
  template <typename Iterable> inline void BuildFrom(const Iterable &src_iterable)
  {
    const int dimensions_count =
      static_cast<int>(std::distance(src_iterable.begin(), src_iterable.end()));
    Resize(dimensions_count);
    int32_t *data = DimsData();
    for (auto it = src_iterable.begin(); it != src_iterable.end(); ++it)
    {
      *data++ = static_cast<int32_t>(*it);
    }
  }

  // Returns the total count of elements, that is the size when flattened into a
  // vector.
  inline static Shape ExtendedShape(int new_shape_size, const Shape &shape)
  {
    return Shape(new_shape_size, shape, 1);
  }

  // Overload for initializer list building.
  inline void BuildFrom(const std::initializer_list<int> init_list)
  {
    BuildFrom<const std::initializer_list<int>>(init_list);
  }

  // Returns the total count of elements (flattened size).
  inline int FlatSize() const
  {
    int buffer_size = 1;
    const int *dims_data = DimsData();
    for (int i = 0; i < _size; i++)
    {
      buffer_size *= dims_data[i];
    }
    return buffer_size;
  }

  bool operator!=(const Shape &comp) const { return !((*this) == comp); }

private:
  // Helper function: initialize dims_ storage based on the number of dimensions.
  inline void initStorage(int dimensions_count)
  {
    assert(dimensions_count >= 0);
    if (dimensions_count <= kMaxSmallSize)
      dims_ = std::array<int32_t, kMaxSmallSize>{};
    else
      dims_ = std::vector<int32_t>(dimensions_count);
  }

  // For use only by ExtendedShape(), written to guarantee (return-value) copy
  // elision in C++17.
  // This creates a shape padded to the desired size with the specified value.
  Shape(int new_shape_size, const Shape &shape, int pad_value) : _size(new_shape_size)
  {
    assert(new_shape_size >= shape.DimensionsCount());
    assert(new_shape_size <= kMaxSmallSize);
    Resize(new_shape_size);
    const int size_increase = new_shape_size - shape.DimensionsCount();
    for (int i = 0; i < size_increase; ++i)
    {
      SetDim(i, pad_value);
    }
    std::memcpy(DimsData() + size_increase, shape.DimsData(),
                sizeof(int32_t) * shape.DimensionsCount());
  }

  int32_t _size;
  // Internal storage: use std::array for shapes with dimensions up to kMaxSmallSize,
  // and std::vector for larger shapes.
  std::variant<std::array<int32_t, kMaxSmallSize>, std::vector<int32_t>> dims_;
};

// Utility functions below.

inline int MatchingDim(const Shape &shape1, int index1, [[maybe_unused]] const Shape &shape2,
                       [[maybe_unused]] int index2)
{
  assert(shape1.Dims(index1) == shape2.Dims(index2));
  return shape1.Dims(index1);
}

template <typename... Args>
int MatchingDim(const Shape &shape1, int index1, [[maybe_unused]] const Shape &shape2,
                [[maybe_unused]] int index2, Args... args)
{
  assert(shape1.Dims(index1) == shape2.Dims(index2));
  return MatchingDim(shape1, index1, args...);
}

inline Shape GetShape(const std::vector<int32_t> &data)
{
  return Shape(static_cast<int>(data.size()), data.data());
}

inline int Offset(const Shape &shape, int i0, int i1, int i2, int i3)
{
  assert(shape.DimensionsCount() == 4);
  const int *dims_data = shape.DimsDataUpTo6D();
  assert(i0 >= 0 && i0 < dims_data[0]);
  assert(i1 >= 0 && i1 < dims_data[1]);
  assert(i2 >= 0 && i2 < dims_data[2]);
  assert(i3 >= 0 && i3 < dims_data[3]);
  return ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3;
}

inline int Offset(const Shape &shape, int i0, int i1, int i2, int i3, int i4)
{
  assert(shape.DimensionsCount() == 5);
  const int *dim = shape.DimsDataUpTo6D();
  assert(i0 >= 0 && i0 < dim[0]);
  assert(i1 >= 0 && i1 < dim[1]);
  assert(i2 >= 0 && i2 < dim[2]);
  assert(i3 >= 0 && i3 < dim[3]);
  assert(i4 >= 0 && i4 < dim[4]);
  return ((((i0 * dim[1] + i1) * dim[2] + i2) * dim[3] + i3) * dim[4]) + i4;
}

inline int Offset(const Shape &shape, int i0, int i1, int i2, int i3, int i4, int i5)
{
  assert(shape.DimensionsCount() == 6);
  const int *dim = shape.DimsDataUpTo6D();
  assert(i0 >= 0 && i0 < dim[0]);
  assert(i1 >= 0 && i1 < dim[1]);
  assert(i2 >= 0 && i2 < dim[2]);
  assert(i3 >= 0 && i3 < dim[3]);
  assert(i4 >= 0 && i4 < dim[4]);
  assert(i5 >= 0 && i5 < dim[5]);
  // clang format off
  return (((((i0 * dim[1] + i1) * dim[2] + i2) * dim[3] + i3) * dim[4]) + i4) * dim[5] + i5;
  // clang format on
}

inline int Offset(const Shape &shape, int *index)
{
  return Offset(shape, index[0], index[1], index[2], index[3], index[4], index[5]);
}

inline int FlatSizeSkipDim(const Shape &shape, int skip_dim)
{
  const int dims_count = shape.DimensionsCount();
  assert(skip_dim >= 0 && skip_dim < dims_count);
  const auto *dims_data = shape.DimsData();
  int flat_size = 1;
  for (int i = 0; i < dims_count; ++i)
  {
    flat_size *= (i == skip_dim) ? 1 : dims_data[i];
  }
  return flat_size;
}

// Flat size calculation, checking that dimensions match with one or more other shapes.
template <typename... Ts> inline bool checkMatching(const Shape &shape, Ts... check_shapes)
{
  auto match = [&shape](const Shape &s) -> bool {
    // Check matching of shapes except the case that both shapes are scalars.
    if (shape.DimensionsCount() > 1 || s.DimensionsCount() > 1 || shape.FlatSize() != 1 ||
        s.FlatSize() != 1)
    {
      if (shape.DimensionsCount() != s.DimensionsCount())
      {
        return false;
      }
      for (int i = 0; i < shape.DimensionsCount(); ++i)
      {
        if (shape.Dims(i) != s.Dims(i))
        {
          return false;
        }
      }
    }
    return true;
  };

  // Apply the lambda to each check shape and combine with &&
  return (match(check_shapes) && ...);
}

struct UNUSED_ALL
{
  template <typename... Args> UNUSED_ALL(Args const &...) {}
};
template <typename... Ts> inline int MatchingFlatSize(const Shape &shape, Ts... check_shapes)
{
  UNUSED_ALL{check_shapes...};
  assert(checkMatching(shape, std::forward<Ts>(check_shapes)...));
  return shape.FlatSize();
}

inline int MatchingFlatSizeSkipDim(const Shape &shape, int skip_dim,
                                   [[maybe_unused]] const Shape &check_shape_0)
{
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i)
  {
    if (i != skip_dim)
    {
      assert(shape.Dims(i) == check_shape_0.Dims(i));
    }
  }
  return FlatSizeSkipDim(shape, skip_dim);
}

inline int MatchingFlatSizeSkipDim(const Shape &shape, int skip_dim,
                                   [[maybe_unused]] const Shape &check_shape_0,
                                   const Shape &check_shape_1)
{
  const int dims_count = shape.DimensionsCount();
  for (int i = 0; i < dims_count; ++i)
  {
    if (i != skip_dim)
    {
      assert(shape.Dims(i) == check_shape_0.Dims(i));
    }
  }
  return MatchingFlatSizeSkipDim(shape, skip_dim, check_shape_1);
}

inline int MatchingElementsSize(const Shape &shape, const Shape &check_shape_0,
                                const Shape &check_shape_1)
{
  const int size_1 = shape.FlatSize();
  [[maybe_unused]] const int size_2 = check_shape_0.FlatSize();
  [[maybe_unused]] const int size_3 = check_shape_1.FlatSize();
  assert(size_1 == size_2);
  assert(size_2 == size_3);
  return size_1;
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_SHAPE_H__
