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

#ifndef _NDARRAY_CXX14_H_
#define _NDARRAY_CXX14_H_

namespace ndarray
{

namespace cxx14
{

template <size_t... Nums> struct index_sequence
{
  using value_type = size_t;

  static constexpr std::size_t size() noexcept { return sizeof...(Nums); }
};

namespace detail
{

template <size_t v, typename Seq> struct _append;

template <size_t v, size_t... Nums> struct _append<v, index_sequence<Nums...>>
{
  using result = index_sequence<Nums..., v>;
};

template <size_t Len> struct make_index_sequence
{
  using result =
    typename detail::_append<Len - 1, typename make_index_sequence<Len - 1>::result>::result;
};

template <> struct make_index_sequence<1>
{
  using result = index_sequence<0>;
};

template <> struct make_index_sequence<0>
{
  using result = index_sequence<>;
};

} // namespace detail

template <size_t Num> using make_index_sequence = typename detail::make_index_sequence<Num>::result;

} // namespace cxx14

} // namespace ndarray

#endif //_NDARRAY_CXX14_H_
