/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __KUMA_DETAILS_LIVE_INTERVAL_SET_H__
#define __KUMA_DETAILS_LIVE_INTERVAL_SET_H__

#include <cstdint>
#include <map>

namespace kuma
{
namespace details
{

struct IntervalMask
{
  uint32_t s;
  uint32_t e;
};

inline IntervalMask mask(uint32_t s, uint32_t e)
{
  IntervalMask mask;

  mask.s = s;
  mask.e = e;

  return mask;
}

class IntervalSet
{
public:
  // [0, len) is live at the beginning
  IntervalSet(uint32_t len = 0xffffffff);

public:
  void insert(const IntervalMask &);

  /**
   * "firstfit(l)" returns the offset of an interval whose length is larger than "l".
   *
   * When multiple intervals meet this condition, "firstfit(l)" chooses the interval
   * with the smallest offset as its name suggests.
   *
   * NOTE This method throws std::runtime_error if fails to find a proper region
   */
  uint32_t firstfit(uint32_t len) const;

private:
  using End = uint32_t;
  using Len = uint32_t;

  // If [e -> l] is in _content, it means that [e - l, e) is a valid interval.
  //
  // INVARIANT
  //
  //  If key m and n (m <= n) are consecutive in _content, "m <= n - _content.at(n)" holds.
  //
  std::map<End, Len> _content;
};

} // namespace details
} // namespace kuma

#endif // __KUMA_DETAILS_LIVE_INTERVAL_SET_H__
