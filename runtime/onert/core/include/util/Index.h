/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_UTIL_INDEX_H__
#define __ONERT_UTIL_INDEX_H__

#include <functional>
#include <limits>
#include <stdint.h>
#include <string>

namespace onert
{
namespace util
{

/**
 * @brief A wrapper class for unsigned integral Index
 *        NOTE : Max value of the underlying type is used as the invalid value
 *
 * @tparam T Underlying type. Must be unsigned integral type otherwise its behavior is undefined.
 * @tparam DummyTag Dummy type to distinguish types with a same underlying type. Using an opaque
 * type is recommended.
 */
template <typename T, typename DummyTag> class Index
{
private:
  static const T UNDEFINED = std::numeric_limits<T>::max();

public:
  /**
   * @brief Construct a new Index object
   */
  explicit Index(void) : _index{UNDEFINED} {}
  /**
   * @brief Construct a new Index object with a value in the underlying type
   *
   * @param o Value in the underlying type
   */
  explicit Index(const T o) : _index{o} {}
  /**
   * @brief Copy Constructor
   *
   * @param o Object to be copied
   */
  Index(const Index &o) = default;

  /**
   * @brief Assign a value in the underlying time
   *
   * @param o Value in the underlying type
   * @return Index& Reference of this pointer
   */
  Index &operator=(const T o)
  {
    _index = o;
    return *this;
  }

  /**
   * @brief Copy assignment operator
   *
   * @param o Object to be copied
   * @return Index& Reference of this pointer
   */
  Index &operator=(const Index &o) = default;

  /**
   * @brief Equality operator
   *
   * @param o The other value in the underlying type to compare
   * @return true if underlying value is the same, false otherwise
   */
  bool operator==(T o) const { return _index == o; }
  /**
   * @brief Equality operator
   *
   * @param o The other object to compare
   * @return true if underlying value is the same, false otherwise
   */
  bool operator==(const Index &o) const { return _index == o._index; }
  /**
   * @brief Inquality operator
   *
   * @param o The other value in the underlying type to compare
   * @return true if underlying value is different, false otherwise
   */
  bool operator!=(T o) const { return !(*this == o); }
  /**
   * @brief Inquality operator
   *
   * @param o The other object to compare
   * @return true if underlying value is different, false otherwise
   */
  bool operator!=(const Index &o) const { return !(*this == o); }

  /**
   * @brief Post increment operator
   *
   * @return Index Index before increment
   */
  Index operator++(int)
  {
    Index temp = *this;
    _index++;
    return temp;
  }

  /**
   * @brief Check whether the value is valid or not
   *
   * @return true if valid, false otherwise
   */
  bool valid() const { return _index != UNDEFINED; }
  /**
   * @brief Check whether the value is undefined
   *
   * @return true if undefined, false otherwise
   */
  bool undefined() const { return _index == UNDEFINED; }
  /**
   * @brief Return underlying value
   *
   * @return T Underlying value
   */
  T value() const { return _index; }

  bool operator<(const Index &I) const { return value() < I.value(); }

  /**
   * @brief Return max index value
   *
   * @return Maximum valid index value
   */
  static T max() { return UNDEFINED - 1; }

private:
  T _index;
};

} // namespace util
} // namespace onert

namespace std
{

template <typename T, typename Tag> struct hash<::onert::util::Index<T, Tag>>
{
  size_t operator()(const ::onert::util::Index<T, Tag> &index) const noexcept
  {
    return hash<T>()(index.value());
  }
};

} // namespace std

#endif // __ONERT_UTIL_INDEX_H__
