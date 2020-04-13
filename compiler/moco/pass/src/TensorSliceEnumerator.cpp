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

#include "TensorSliceEnumerator.h"

#include <cassert>

namespace moco
{

void TensorSliceEnumerator::shape(loco::TensorShape &s)
{
  _shape_in = s;
  _rank_in = _shape_in.rank();
}

void TensorSliceEnumerator::increment(uint32_t r)
{
  if (_cursor.at(r) < _boundary.at(r))
    _cursor.at(r) = _cursor.at(r) + 1;
  else
  {
    if (r > 0)
    {
      _cursor.at(r) = _begin[r];
      increment(r - 1);
    }
    else
    {
      // reached to the end
    }
  }
}

void TensorSliceEnumerator::start(void)
{
  auto rank = _rank_in;

  _cursor.resize(rank);
  _boundary.resize(rank);
  for (uint32_t r = 0; r < rank; ++r)
  {
    _cursor.at(r) = _begin[r];
    _boundary.at(r) = _end[r];
  }
}

bool TensorSliceEnumerator::valid(void)
{
  auto rank = _rank_in;
  for (uint32_t r = 0; r < rank; ++r)
  {
    if (_cursor.at(r) >= _boundary.at(r))
      return false;
  }
  return true;
}

void TensorSliceEnumerator::advance(void)
{
  uint32_t r = _rank_in - 1;
  increment(r);
}

uint32_t TensorSliceEnumerator::cursor(uint32_t rank) const
{
  assert(rank < _rank_in);
  return _cursor.at(rank);
}

} // namespace moco
