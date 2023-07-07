/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_RUY_SHAPE_H__
#define __NNFW_RUY_SHAPE_H__

#include <algorithm>
#include <cstring>
#include <cassert>
#include <vector>

#define UNUSED_RELEASE(a) (void)(a)

namespace nnfw
{
namespace ruy
{

class Shape
{
public:
  // Shapes with dimensions up to 5 are stored directly in the structure, while
  // larger shapes are separately allocated.
  static constexpr int kMaxSmallSize = 5;

  Shape &operator=(Shape const &) = delete;

  Shape() : _size(0) {}

  explicit Shape(int dimensions_count) : _size(dimensions_count)
  {
    if (dimensions_count > kMaxSmallSize)
    {
      _dims_pointer = new int32_t[dimensions_count];
    }
  }

  Shape(int shape_size, int32_t value) : _size(0)
  {
    Resize(shape_size);
    for (int i = 0; i < shape_size; ++i)
    {
      SetDim(i, value);
    }
  }

  Shape(int dimensions_count, const int32_t *dims_data) : _size(0)
  {
    ReplaceWith(dimensions_count, dims_data);
  }

  Shape(const std::initializer_list<int> init_list) : _size(0) { BuildFrom(init_list); }

  // Avoid using this constructor.  We should be able to delete it when C++17
  // rolls out.
  Shape(Shape const &other) : _size(other.DimensionsCount())
  {
    if (_size > kMaxSmallSize)
    {
      _dims_pointer = new int32_t[_size];
    }
    std::memcpy(DimsData(), other.DimsData(), sizeof(int32_t) * _size);
  }

  bool operator==(const Shape &comp) const
  {
    return this->_size == comp._size &&
           std::memcmp(DimsData(), comp.DimsData(), _size * sizeof(int32_t)) == 0;
  }

  ~Shape()
  {
    if (_size > kMaxSmallSize)
    {
      delete[] _dims_pointer;
    }
  }

  inline int32_t DimensionsCount() const { return _size; }
  inline int32_t Dims(int i) const
  {
    assert(i >= 0);
    assert(i < _size);
    return _size > kMaxSmallSize ? _dims_pointer[i] : _dims[i];
  }
  inline void SetDim(int i, int32_t val)
  {
    assert(i >= 0);
    assert(i < _size);
    if (_size > kMaxSmallSize)
    {
      _dims_pointer[i] = val;
    }
    else
    {
      _dims[i] = val;
    }
  }

  inline int32_t *DimsData() { return _size > kMaxSmallSize ? _dims_pointer : _dims; }
  inline const int32_t *DimsData() const { return _size > kMaxSmallSize ? _dims_pointer : _dims; }
  // The caller must ensure that the shape is no bigger than 4-D.
  inline const int32_t *DimsDataUpTo4D() const { return _dims; }

  inline void Resize(int dimensions_count)
  {
    if (_size > kMaxSmallSize)
    {
      delete[] _dims_pointer;
    }
    _size = dimensions_count;
    if (dimensions_count > kMaxSmallSize)
    {
      _dims_pointer = new int32_t[dimensions_count];
    }
  }

  inline void ReplaceWith(int dimensions_count, const int32_t *dims_data)
  {
    Resize(dimensions_count);
    int32_t *dst_dims = DimsData();
    std::memcpy(dst_dims, dims_data, dimensions_count * sizeof(int32_t));
  }

  inline void ReplaceWith(const Shape &other)
  {
    ReplaceWith(other.DimensionsCount(), other.DimsData());
  }

  inline void ReplaceWith(Shape &&other)
  {
    Resize(0);
    std::swap(_size, other._size);
    if (_size <= kMaxSmallSize)
      std::copy(other._dims, other._dims + kMaxSmallSize, _dims);
    else
      _dims_pointer = other._dims_pointer;
  }

  template <typename T> inline void BuildFrom(const T &src_iterable)
  {
    const int dimensions_count = std::distance(src_iterable.begin(), src_iterable.end());
    Resize(dimensions_count);
    int32_t *data = DimsData();
    for (auto &&it : src_iterable)
    {
      *data = it;
      ++data;
    }
  }

  // This will probably be factored out. Old code made substantial use of 4-D
  // shapes, and so this function is used to extend smaller shapes. Note that
  // (a) as Dims<4>-dependent code is eliminated, the reliance on this should be
  // reduced, and (b) some kernels are stricly 4-D, but then the shapes of their
  // inputs should already be 4-D, so this function should not be needed.
  inline static Shape ExtendedShape(int new_shape_size, const Shape &shape)
  {
    return Shape(new_shape_size, shape, 1);
  }

  inline void BuildFrom(const std::initializer_list<int> init_list)
  {
    BuildFrom<const std::initializer_list<int>>(init_list);
  }

  // Returns the total count of elements, that is the size when flattened into a
  // vector.
  inline int FlatSize() const
  {
    int buffer_size = 1;
    const int *dims_data = DimsData();
    for (int i = 0; i < _size; i++)
    {
      const int dim = dims_data[i];
      assert(dim >= 1);
      buffer_size *= dim;
    }
    return buffer_size;
  }

  bool operator!=(const Shape &comp) const { return !((*this) == comp); }

private:
  // For use only by ExtendedShape(), written to guarantee (return-value) copy
  // elision in C++17.
  // This creates a shape padded to the desired size with the specified value.
  Shape(int new_shape_size, const Shape &shape, int pad_value) : _size(0)
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
  union {
    int32_t _dims[kMaxSmallSize];
    int32_t *_dims_pointer{nullptr};
  };
};

inline int MatchingDim(const Shape &shape1, int index1, const Shape &shape2, int index2)
{
  UNUSED_RELEASE(shape2);
  UNUSED_RELEASE(index2);
  assert(shape1.Dims(index1) == shape2.Dims(index2));
  return shape1.Dims(index1);
}

template <typename... Args>
int MatchingDim(const Shape &shape1, int index1, const Shape &shape2, int index2, Args... args)
{
  assert(shape1.Dims(index1) == shape2.Dims(index2));
  UNUSED_RELEASE(shape2);
  UNUSED_RELEASE(index2);
  return MatchingDim(shape1, index1, args...);
}

inline Shape GetShape(const std::vector<int32_t> &data) { return Shape(data.size(), data.data()); }

inline int Offset(const Shape &shape, int i0, int i1, int i2, int i3)
{
  assert(shape.DimensionsCount() == 4);
  const int *dims_data = shape.DimsDataUpTo4D();
  assert(i0 >= 0 && i0 < dims_data[0]);
  assert(i1 >= 0 && i1 < dims_data[1]);
  assert(i2 >= 0 && i2 < dims_data[2]);
  assert(i3 >= 0 && i3 < dims_data[3]);
  return ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3;
}

inline int Offset(const Shape &shape, int *index)
{
  return Offset(shape, index[0], index[1], index[2], index[3]);
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

// Flat size calculation, checking that dimensions match with one or more other
// arrays.
template <typename... Ts> inline bool checkMatching(const Shape &shape, Ts... check_shapes)
{
  const Shape check_shapes_array[sizeof...(Ts)] = {std::forward<Ts>(check_shapes)...};
  for (const auto &check_shape : check_shapes_array)
  {
    // Check matching of shapes except the case of that two shapes can be scalar
    if (shape.DimensionsCount() > 1 || check_shape.DimensionsCount() > 1 || shape.FlatSize() != 1 ||
        check_shape.FlatSize() != 1)
    {
      if (shape.DimensionsCount() != check_shape.DimensionsCount())
      {
        return false;
      }
      for (int i = 0; i < shape.DimensionsCount(); ++i)
      {
        if (shape.Dims(i) != check_shape.Dims(i))
        {
          return false;
        }
      }
    }
  }
  return true;
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

inline int MatchingFlatSizeSkipDim(const Shape &shape, int skip_dim, const Shape &check_shape_0)
{
  UNUSED_RELEASE(check_shape_0);
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

inline int MatchingFlatSizeSkipDim(const Shape &shape, int skip_dim, const Shape &check_shape_0,
                                   const Shape &check_shape_1)
{
  UNUSED_RELEASE(check_shape_0);
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
  const int size_2 = check_shape_0.FlatSize();
  const int size_3 = check_shape_1.FlatSize();
  assert(size_1 == size_2);
  assert(size_2 == size_3);
  UNUSED_RELEASE(size_2);
  UNUSED_RELEASE(size_3);
  return size_1;
}

} // namespace ruy
} // namespace nnfw

#endif // __NNFW_RUY_SHAPE_H__
