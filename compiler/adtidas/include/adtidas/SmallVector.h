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

#ifndef _ADTIDAS_SMALL_VECTOR_H_
#define _ADTIDAS_SMALL_VECTOR_H_

#include <cassert>
#include <iterator>
#include <initializer_list>

namespace adt
{

/**
 * @brief vector with cheap memory allocation
 * @tparam T type of elements
 * @tparam Capacity maximum number of elements
 * @note much like std::array, but tracks number of used elements. Stored in stack
 */
template <typename T, size_t Capacity> class small_vector
{
public:
  using value_type = T;
  using reference = T &;
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;

  template <typename It> small_vector(It begin, It end) : _size(std::distance(begin, end))
  {
    assert(_size <= Capacity);
    std::copy(begin, end, this->begin());
  }

  explicit small_vector(size_t size, value_type initializer = value_type()) : _size(size)
  {
    assert(_size <= Capacity);
    std::fill(begin(), end(), initializer);
  }

  explicit small_vector() : _size(0) {}

  small_vector(std::initializer_list<value_type> l) : _size(l.size())
  {
    assert(_size <= Capacity);
    std::copy(std::begin(l), std::end(l), begin());
  }

  /**
   * @return current size
   */
  inline size_t size() const noexcept { return _size; }

  /**
   * @return maximum number of elements this vector can hold
   */
  constexpr size_t capacity() const { return Capacity; }

  /**
   * @brief resize to given new size
   * @note if new size is greater than current size, new elements are default-initialized
   */
  void resize(size_t new_size) noexcept
  {
    assert(new_size <= Capacity);
    if (new_size > _size)
    {
      std::fill(_storage + _size, _storage + new_size, T());
    }
    _size = new_size;
  }

  /**
   * @return reference to the element at position idx
   */
  inline reference operator[](size_t idx) noexcept
  {
    assert(idx < _size);
    return _storage[idx];
  }

  /**
   * @return value of element at position idx
   */
  inline constexpr value_type operator[](size_t idx) const noexcept
  {
    // assert on the same line since c++11 does not allow multi-line constexpr functions
    return assert(idx < _size), _storage[idx];
  }

  inline iterator begin() noexcept { return std::begin(_storage); }
  inline iterator end() noexcept { return _storage + _size; }

  inline reverse_iterator rbegin() noexcept { return reverse_iterator{end()}; }
  inline reverse_iterator rend() noexcept { return reverse_iterator{begin()}; }

  // const overloads
  inline const_iterator begin() const noexcept { return std::begin(_storage); }
  inline const_iterator end() const noexcept { return _storage + _size; }

  inline const_reverse_iterator rbegin() const noexcept { return reverse_iterator{end()}; }
  inline const_reverse_iterator rend() const noexcept { return reverse_iterator{begin()}; }

  inline void push_back(const value_type &e) noexcept
  {
    assert(_size < Capacity);
    _storage[_size++] = e;
  }

  inline void push_back(value_type &&e) noexcept
  {
    assert(_size < Capacity);
    _storage[_size++] = std::move(e);
  }

private:
  size_t _size;
  value_type _storage[Capacity]{};
};

template <typename T, size_t LCapacity, size_t RCapacity>
bool operator==(const small_vector<T, LCapacity> &lhs, const small_vector<T, RCapacity> &rhs)
{
  if (lhs.size() != rhs.size())
  {
    return false;
  }

  bool equal = true;
  size_t end = lhs.size();
  for (size_t i = 0; i < end; ++i)
  {
    equal &= (lhs[i] == rhs[i]);
  }

  return equal;
}

} // namespace adt

#endif //_ADTIDAS_SMALL_VECTOR_H_
