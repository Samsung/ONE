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

#ifndef __ONERT_UTIL_OBJECT_MANAGER_H__
#define __ONERT_UTIL_OBJECT_MANAGER_H__

#include "util/logging.h"

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <unordered_map>

namespace onert
{
namespace util
{

/**
 * @brief Class that owns objects and maps them with indices as a handle for them
 *
 */
template <typename Index, typename Object> class ObjectManager
{
public:
  ObjectManager() : _next_index{0u} {}

public:
  /**
   * @brief Create an object with args and put it in the container with a newly assigned @c Index
   *
   * @param[in] args Arguments for creating Operand object
   * @return Created index that is associated to the object if successful, Undefined index otherwise
   */
  template <class... Args> Index emplace(Args &&...args)
  {
    auto index = generateIndex();
    if (!index.valid())
      return index;
    _objects.emplace(index, std::make_unique<Object>(std::forward<Args>(args)...));
    return index;
  }

  /**
   * @brief Put the object in the container with given index.
   *
   * It fails when the given index is already taken or @c index is Undefined.
   *
   * @param[in] object Object to be pushed
   * @param[in] index Index associated with the object
   * @return @c index if successful, an Undefined index otherwise
   */
  Index push(std::unique_ptr<Object> &&object, Index index)
  {
    auto gen_index = tryIndex(index);
    if (gen_index.valid())
      _objects.emplace(gen_index, std::move(object));
    return gen_index;
  }
  /**
   * @brief Put the object in the container with a newly assigned index.
   *
   * It fails when it cannot generate a valid index.
   *
   * @param[in] object Object to be pushed
   * @return The newly assigned index if successful, an Undefined index otherwise
   */
  Index push(std::unique_ptr<Object> &&object)
  {
    auto gen_index = generateIndex();
    if (gen_index.valid())
      _objects.emplace(gen_index, std::move(object));
    return gen_index;
  }
  /**
   * @brief Set the object in the container with given index.
   *
   * If the index is Undefined, it will fail.
   * If the index is already taken, it will overwrite the content.
   *
   * @param[in] object Object to be pushed
   * @param[in] index Index associated with the object
   * @return @c index if successful, an Undefined index otherwise
   */
  Index set(Index index, std::unique_ptr<Object> &&object)
  {
    if (index.valid())
      _objects[index] = std::move(object);
    return index;
  }
  /**
   * @brief Remove the object that is associated with the given index
   *
   * @param[in] index Index of the object to be removed
   * @return N/A
   */
  void remove(const Index &index) { _objects.erase(index); }

  /**
   * @brief Get the object that is associated with the given index
   *
   * If such object does not exist, it will throw @c std::out_of_range
   *
   * @param[in] index Index of the object to be returned
   * @return Object
   */
  const Object &at(const Index &index) const { return *(_objects.at(index)); }
  /**
   * @brief Get the object that is associated with the given index
   *
   * If such object does not exist, it will throw @c std::out_of_range
   *
   * @param[in] index Index of the object to be returned
   * @return Object
   */
  Object &at(const Index &index) { return *(_objects.at(index)); }
  /**
   * @brief Get the object that is associated with the given index
   *
   * If such object does not exist, it will return `nullptr`
   *
   * @param[in] index Index of the object to be returned
   * @return Object
   */
  const Object *getRawPtr(const Index &index) const
  {
    auto itr = _objects.find(index);
    if (itr == _objects.end())
      return nullptr;
    else
    {
      assert(itr->second != nullptr);
      return itr->second.get();
    }
  }
  /**
   * @brief Get the object that is associated with the given index
   *
   * If such object does not exist, it will return `nullptr`
   *
   * @param[in] index Index of the object to be returned
   * @return Object The found object
   */
  Object *getRawPtr(const Index &index)
  {
    return const_cast<Object *>(
      const_cast<const ObjectManager<Index, Object> *>(this)->getRawPtr(index));
  }
  /**
   * @brief Get the object that is associated with the given index
   *
   * @param[in] index Index of the object to be returned
   * @return true if such entry exists otherwise false
   */
  bool exist(const Index &index) const
  {
    auto it = _objects.find(index);
    return it != _objects.end();
  }
  /**
   * @brief Return the number of objects that the manager contains
   *
   * @return size_t Number of objects
   */
  size_t size() const { return _objects.size(); }
  /**
   * @brief Iterate over the container with given function
   *
   * @param[in] fn Function to be run for every container entry
   * @return N/A
   */
  void iterate(const std::function<void(const Index &, const Object &)> &fn) const
  {
    for (const auto &e : _objects)
    {
      fn(e.first, *e.second);
    }
  }
  /**
   * @brief Iterate over the container with given function
   *
   * @param[in] fn Function to be run for every container entry
   * @return N/A
   */
  void iterate(const std::function<void(const Index &, Object &)> &fn)
  {
    // TODO Remove this workaround
    // This implementation is a workaround in case of adding operands while iteration
    std::list<Index> l;

    for (const auto &e : _objects)
    {
      l.push_back(e.first);
    }

    for (const auto &index : l)
    {
      fn(index, *_objects[index]);
    }
  }

private:
  // Try assigning the given index
  Index tryIndex(Index index)
  {
    if (!index.valid())
      return index;
    if (_objects.find(index) == _objects.end())
    {
      // If the given index does not exist, update the next index and return the index
      if (index.value() >= _next_index)
        _next_index = index.value() + 1;
      return index;
    }
    else
    {
      // If the given index exists already, return a non-valid index
      return Index{};
    }
  }

  // Generate a new index with `_next_index`
  Index generateIndex()
  {
    // No need to check if there is an entry with _next_index since
    // _next_index is always ("the highest index in the object map" + 1)
    if (Index{_next_index}.valid())
      return Index{_next_index++};
    else
      return Index{};
  }

protected:
  std::unordered_map<Index, std::unique_ptr<Object>> _objects;
  uint32_t _next_index;
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_OBJECT_MANAGER_H__
