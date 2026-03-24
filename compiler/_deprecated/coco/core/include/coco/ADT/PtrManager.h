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

#ifndef __COCO_ADT_PTR_MANAGER_H__
#define __COCO_ADT_PTR_MANAGER_H__

#include <vector>

#include <memory>
#include <stdexcept>

namespace coco
{

template <typename T> class PtrManager
{
public:
  /// @brief Return the number of managed objects
  uint32_t size(void) const { return _ptrs.size(); }

public:
  T *at(uint32_t n) const { return _ptrs.at(n).get(); }

protected:
  template <typename U> U *take(std::unique_ptr<U> &&o)
  {
    auto res = o.get();
    _ptrs.emplace_back(std::move(o));
    return res;
  }

protected:
  std::unique_ptr<T> release(T *ptr)
  {
    for (auto it = _ptrs.begin(); it != _ptrs.end(); ++it)
    {
      if (it->get() == ptr)
      {
        std::unique_ptr<T> res = std::move(*it);
        _ptrs.erase(it);
        return res;
      }
    }

    throw std::invalid_argument{"ptr"};
  }

private:
  std::vector<std::unique_ptr<T>> _ptrs;
};

} // namespace coco

#endif // __COCO_ADT_PTR_MANAGER_H__
