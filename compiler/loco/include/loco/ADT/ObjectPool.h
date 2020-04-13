/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_ADT_OBJECT_POOL_H__
#define __LOCO_ADT_OBJECT_POOL_H__

#include <algorithm>
#include <memory>
#include <vector>

namespace loco
{

/**
 * @brief Object Pool
 * @note ObjectPool owns registered objects.
 */
template <typename T> class ObjectPool
{
public:
  virtual ~ObjectPool() = default;

public:
  /// @brief Return the number of objects
  uint32_t size(void) const { return _pool.size(); }

  /// @brief Access N-th object
  T *at(uint32_t n) const { return _pool.at(n).get(); }

protected:
  /// @brief Take the ownership of a given object and returns its raw pointer
  template <typename U> U *take(std::unique_ptr<U> &&o)
  {
    auto res = o.get();
    _pool.emplace_back(std::move(o));
    return res;
  }

  /**
   * @brief Erase an object from the pool
   *
   * erase(p) returns false if p does not belong to this object pool.
   */
  bool erase(T *ptr)
  {
    auto pred = [ptr](const std::unique_ptr<T> &o) { return o.get() == ptr; };
    auto it = std::find_if(_pool.begin(), _pool.end(), pred);

    if (it == _pool.end())
    {
      return false;
    }

    _pool.erase(it);
    return true;
  }

private:
  std::vector<std::unique_ptr<T>> _pool;
};

} // namespace loco

#endif // __LOCO_ADT_OBJECT_POOL_H__
