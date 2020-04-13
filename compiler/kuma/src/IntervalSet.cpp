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

#include "IntervalSet.h"

#include <cassert>
#include <stdexcept>

namespace kuma
{
namespace details
{

IntervalSet::IntervalSet(uint32_t len)
{
  // Update _content
  _content[len] = len;
}

void IntervalSet::insert(const IntervalMask &m)
{
  auto s = m.s;
  auto e = m.e;

  assert(s <= e);

  if (s == e)
  {
    // Empty region, nothing to do
    return;
  }

  // lower_bound() returns an iterator to the first element not less than the given key
  auto lb = _content.lower_bound(s);

  // NOTE 1. "lower_bound" ensures "prev_s < s <= curr_e"
  // NOTE 2. "e" points to somewhere after "s"
  auto curr_s = lb->first - lb->second;
  auto curr_e = lb->first;

  if (curr_s < s)
  {
    // Split the current interval
    _content[s] = s - curr_s;
    // NOTE The invariant over "_content" is temporarily broken here.
  }

  if (e < curr_e)
  {
    // Adjust the current interval
    _content[curr_e] = curr_e - e;
  }
  else
  {
    // Remove the current interval
    _content.erase(curr_e);
    // Check the next interval (e > curr_e)
    //
    // TODO Remove this recursive call (to prevent stack overflow issue)
    insert(mask(curr_e, e));
  }
}

uint32_t IntervalSet::firstfit(uint32_t len) const
{
  for (auto it = _content.begin(); it != _content.end(); ++it)
  {
    if (it->second >= len)
    {
      // Got it! This interval is larger than "len".
      return it->first - it->second;
    }
  }

  throw std::runtime_error{"infeasible"};
}

} // namespace details
} // namespace kuma
