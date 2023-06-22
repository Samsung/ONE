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

/**
 * @file     Set.h
 * @brief    This file contains onert::util::Set class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __ONERT_UTIL_SET_H__
#define __ONERT_UTIL_SET_H__

#include <cassert>
#include <unordered_set>

namespace onert
{
namespace util
{

/**
 * @brief Class for set of custom element
 & @tparam Element  Key type of Set
 */
template <typename Element> class Set
{
public:
  /**
   * @brief Construct default Set object.
   */
  Set() = default;
  /**
   * @brief Construct Set object by copy semantics.
   */
  Set(const Set<Element> &) = default;
  /**
   * @brief Construct move Set object by move semantics.
   */
  Set(Set<Element> &&) = default;

public:
  /**
   * @brief copy assignment operator
   */
  Set<Element> &operator=(const Set<Element> &) = default;
  /**
   * @brief move assignment operator
   */
  Set<Element> &operator=(Set<Element> &&) = default;

public:
  /**
   * @brief Add a given element to the set
   *
   * @param e Element added
   */
  void add(const Element &e) { _set.insert(e); }
  /**
   * @brief remove a given element from the set
   *
   * @param e Element removed
   */
  void remove(const Element &e) { _set.erase(e); }
  /**
   * @brief Get size of the set
   *
   * @return The size of the set
   */
  uint32_t size() const { return static_cast<uint32_t>(_set.size()); }
  /**
   * @brief Get whether the set is empty
   *
   * @return Whether the set is empty
   */
  bool empty() const { return _set.empty(); }
  /**
   * @brief Get whether a given element exists in the set
   *
   * @param e A given element
   *
   * @return Whether a given element exists in the set
   */
  bool contains(const Element &e) const { return _set.find(e) != _set.end(); }
  /**
   * @brief Get first element of the set
   *
   * @return first element of the set
   */
  const Element &getOnlyElement() const
  {
    assert(_set.size() == 1u);
    return *_set.begin();
  }

public:
  /**
   * @brief operator overloading function for `|`
   *
   * @return A set with two sets combined
   */
  Set<Element> operator|(const Set<Element> &other) const // Union
  {
    auto ret = *this;
    for (auto e : other)
    {
      ret.add(e);
    }
    return ret;
  }
  /**
   * @brief operator overloading function for `&`
   *
   * @return A set of elements that overlap in two sets
   */
  Set<Element> operator&(const Set<Element> &other) const // Intersect
  {
    Set<Element> ret;
    for (auto e : other)
    {
      if (contains(e))
      {
        ret.add(e);
      }
    }
    return ret;
  }
  /**
   * @brief operator overloading function for `-`
   *
   * @return A set of subtracted from another set
   */
  Set<Element> operator-(const Set<Element> &other) const // Minus
  {
    auto ret = *this;
    for (auto e : other)
    {
      ret.remove(e);
    }
    return ret;
  }

public:
  /**
   * @brief begin() of const_iterator for this class
   *
   * @return The first iterator of the set
   */
  typename std::unordered_set<Element>::const_iterator begin() const { return _set.begin(); }
  /**
   * @brief end() of const_iterator for this class
   *
   * @return The last iterator of the set
   */
  typename std::unordered_set<Element>::const_iterator end() const { return _set.end(); }

private:
  std::unordered_set<Element> _set;
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_SET_H__
