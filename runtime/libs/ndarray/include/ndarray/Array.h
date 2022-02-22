/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _NDARRAY_ARRAY_H_
#define _NDARRAY_ARRAY_H_

#include "Common.h"

#include "ContiguousSpan.h"
#include "Shape.h"

#if __cplusplus < 201402L
#include "detail/cxx14.h" //integer_sequence and make_index_dequence definitions
#else
#include <utility>
#endif

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <array>
#include <tuple>
#include <cstddef>

namespace ndarray
{

// there is no index_sequence before c++14
#if __cplusplus < 201402L

template <size_t... Nums> using index_sequence = cxx14::index_sequence<Nums...>;

template <size_t Num> using make_index_sequence = cxx14::make_index_sequence<Num>;

#else

template <size_t... Nums> using index_sequence = std::index_sequence<Nums...>;

template <size_t _Num> using make_index_sequence = std::make_index_sequence<_Num>;

#endif //__cplusplus < 201402L

struct Strides
{
  explicit Strides(Shape s) : _strides{} { fillStrides(s); }

  int operator[](size_t idx) const noexcept { return _strides[idx]; }

  // since we don't have c++14 fold expression
  template <typename Seq, typename... Ts> struct _calc_offset;

  template <size_t Num, size_t... Nums, typename T, typename... Ts>
  struct _calc_offset<index_sequence<Num, Nums...>, T, Ts...>
  {
    static constexpr size_t get(const std::array<int, 8> &strides, int x, Ts... xs)
    {
      return _calc_offset<index_sequence<Nums...>, Ts...>::get(strides, xs...) +
             x * std::get<Num>(strides);
    }
  };

  template <size_t Num, typename T> struct _calc_offset<index_sequence<Num>, T>
  {
    static constexpr size_t get(const std::array<int, 8> &strides, int x)
    {
      return x * std::get<Num>(strides);
    }
  };

  template <typename Seq, typename... Ts> constexpr size_t offset(Seq, Ts... x) const noexcept
  {
    // return ( 0 + ... + (std::get<Nums>(_strides) * x)); in c++14
    return _calc_offset<Seq, Ts...>::get(_strides, x...);
  }

private:
  void fillStrides(const Shape &s) noexcept
  {
    int rank = s.rank();
    _strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
    {
      _strides[d] = _strides[d + 1] * s.dim(d + 1);
    }
  }

  std::array<int, NDARRAY_MAX_DIMENSION_COUNT> _strides;
};

template <typename T> class Array
{
public:
  Array(T *data, Shape shape) noexcept : _data(data), _shape(shape), _strides(shape) {}

  Array(const Array &) = delete;

  Array(Array &&a) noexcept : _data(a._data), _shape(a._shape), _strides(a._strides)
  {
    a._data = nullptr;
  }

  template <typename... Ts> T &at(Ts... x) const noexcept { return _at(static_cast<size_t>(x)...); }

  /**
   * @brief returns last dimension as ContigniousSpan
   * @param x indices of slice to take. See tests for usage details
   * @return slice at given position
   */
  template <typename... Ts> ContiguousSpan<T, std::is_const<T>::value> slice(Ts... x) noexcept
  {
    assert(sizeof...(Ts) == _shape.rank() - 1);
    return {&at(x..., 0ul), _shape.dim(_shape.rank() - 1)};
  }

  /**
   * @brief returns last dimension as ContigniousSpan
   * @param x indices of slice to take. See tests for usage details
   * @return slice at given position
   */
  template <typename... Ts> ContiguousSpan<T, true> slice(Ts... x) const noexcept
  {
    assert(sizeof...(Ts) == _shape.rank() - 1);
    return {&at(x..., 0ul), _shape.dim(_shape.rank() - 1)};
  }

  ContiguousSpan<T, std::is_const<T>::value> flat() noexcept
  {
    return {_data, _shape.element_count()};
  }

  ContiguousSpan<T, true> flat() const noexcept { return {_data, _shape.element_count()}; }

  const Shape &shape() const noexcept { return _shape; }

private:
  template <typename... Ts> T &_at(Ts... x) const noexcept
  {
    assert(sizeof...(x) == _shape.rank());
    using Indices = make_index_sequence<sizeof...(Ts)>;
    return _data[offset(Indices{}, x...)];
  }

  template <typename... Ts, size_t... Nums>
  size_t offset(index_sequence<Nums...> seq, Ts... x) const noexcept
  {
    static_assert(
      sizeof...(Ts) == sizeof...(Nums),
      "Sanity check failed. Generated index sequence size is not equal to argument count");

    return _strides.offset(seq, x...);
  }

  T *_data;
  Shape _shape;
  Strides _strides;
};

template <typename To, typename From> Array<To> array_cast(Array<From> &&from, Shape newShape)
{
  assert(from.shape().element_count() / (sizeof(To) / sizeof(From)) == newShape.element_count());
  return Array<To>(reinterpret_cast<To *>(from.flat().data()), newShape);
}

template <typename To, typename From>
Array<const To> array_cast(const Array<From> &from, Shape newShape)
{
  assert(from.shape().element_count() / (sizeof(To) / sizeof(From)) == newShape.element_count());
  return Array<To>(reinterpret_cast<const To *>(from.flat().data()), newShape);
}

#ifndef NDARRAY_INLINE_TEMPLATES

extern template class Array<float>;
extern template class Array<int32_t>;
extern template class Array<uint32_t>;
extern template class Array<uint8_t>;

#endif // NDARRAY_INLINE_TEMPLATES

} // namespace ndarray

#endif //_NDARRAY_ARRAY_H_
