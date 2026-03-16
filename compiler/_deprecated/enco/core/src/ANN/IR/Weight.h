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

#ifndef __ANN_IR_WEIGHT_H__
#define __ANN_IR_WEIGHT_H__

#include <vector>

#include <cstdint>
#include <type_traits>

namespace ann
{

class Weight
{
public:
  const uint8_t *base(void) const { return _buffer.data(); }
  uint32_t size(void) const { return _buffer.size(); }

public:
  template <typename T> void fill(const T &value)
  {
    static_assert(std::is_arithmetic<T>::value, "T should be arithmetic");
    _buffer.clear();

    auto arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(T); ++b)
    {
      _buffer.emplace_back(arr[b]);
    }
  }

  template <typename It> void fill(It beg, It end)
  {
    _buffer.clear();

    for (auto it = beg; it != end; ++it)
    {
      const auto value = *it;
      auto arr = reinterpret_cast<const uint8_t *>(&value);

      for (uint32_t b = 0; b < sizeof(value); ++b)
      {
        _buffer.emplace_back(arr[b]);
      }
    }
  }

private:
  std::vector<uint8_t> _buffer;
};

} // namespace ann

#endif // __ANN_IR_WEIGHT_H__
