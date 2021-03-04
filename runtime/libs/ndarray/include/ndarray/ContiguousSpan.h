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

#ifndef _NDARRAY_CONTIGNIOUS_SPAN_H_
#define _NDARRAY_CONTIGNIOUS_SPAN_H_

#include <type_traits>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>

namespace ndarray
{

template <typename T, bool isConst = false> class ContiguousSpan
{
public:
  using pointer_type = typename std::conditional<isConst, const T *, T *>::type;
  using reference_type = typename std::conditional<isConst, const T &, T &>::type;
  using iterator_type = pointer_type;

  ContiguousSpan(pointer_type data, size_t len) noexcept : _data(data), _len(len) {}

  template <typename It>
  explicit ContiguousSpan(It first, It last) noexcept
    : _data(&*first), _len(std::distance(first, last))
  {
  }

  ContiguousSpan(const ContiguousSpan &) = delete;

  ContiguousSpan(ContiguousSpan &&s) noexcept : _data(s._data), _len(s._len) { s._data = nullptr; }

  operator ContiguousSpan<T, true>() { return ContiguousSpan<T, true>{_data, _len}; }

  reference_type operator[](size_t idx) const noexcept { return _data[idx]; }

  reference_type at(size_t idx) const noexcept { return _data[idx]; }

  ContiguousSpan<T, isConst> offset(size_t offset)
  {
    assert(offset <= _len);
    return {_data + offset, _len - offset};
  }

  template <typename From, bool _ = isConst>
  typename std::enable_if<!_, void>::type assign(const From &f) noexcept
  {
    assignFrom(std::begin(f), std::end(f));
  }

  template <typename U, bool _ = isConst>
  typename std::enable_if<!_, ContiguousSpan &>::type
  operator=(std::initializer_list<U> list) noexcept
  {
    assignFrom(std::begin(list), std::end(list));
    return *this;
  }

  template <typename It, bool _ = isConst>
  typename std::enable_if<!_, void>::type assignFrom(It first, It last) noexcept
  {
    std::copy(first, last, begin());
  }

  size_t size() const { return _len; }

  iterator_type begin() const { return iterator_type{_data}; }

  iterator_type end() const { return iterator_type{_data + _len}; }

  pointer_type data() { return _data; }

private:
  pointer_type _data;
  size_t _len;
};

#ifndef NDARRAY_INLINE_TEMPLATES

extern template class ContiguousSpan<float, true>;
extern template class ContiguousSpan<float, false>;
extern template class ContiguousSpan<int32_t, true>;
extern template class ContiguousSpan<int32_t, false>;
extern template class ContiguousSpan<uint32_t, true>;
extern template class ContiguousSpan<uint32_t, false>;
extern template class ContiguousSpan<uint8_t, true>;
extern template class ContiguousSpan<uint8_t, false>;

#endif // NDARRAY_INLINE_TEMPLATES

} // namespace ndarray

#endif //_NDARRAY_CONTIGNIOUS_SPAN_H_
