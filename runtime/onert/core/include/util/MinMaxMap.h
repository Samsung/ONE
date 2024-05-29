/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_UTIL_MINMAX_MAP_H_
#define __ONERT_UTIL_MINMAX_MAP_H_

#include <unordered_map>
#include <utility>

namespace onert
{
namespace util
{

template <typename N, typename Hash = std::hash<N>> class MinMaxMap
{
  struct MinMaxPair
  {
    float data[2]; // [0] = min, [1] = max
  };

public:
  void append(N node, float min, float max) { _minmax_map[node] = {min, max}; }
  auto begin() const { return _minmax_map.begin(); }
  auto end() const { return _minmax_map.end(); }
  auto size() const { return _minmax_map.size(); }

private:
  std::unordered_map<N, MinMaxPair, Hash> _minmax_map;
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_MINMAX_MAP_H_
