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

#include "TensorPackEnumerator.h"

#include <cassert>

namespace moco
{

void TensorPackEnumerator::shape(const loco::TensorShape &si, const loco::TensorShape &so)
{
  _shape_inp = si;
  _shape_out = so;

  assert(_shape_inp.rank() + 1 == _shape_out.rank());

  _rank_out = _shape_out.rank();
}

void TensorPackEnumerator::increment(uint32_t r)
{
  _cursor_out.at(r) = _cursor_out.at(r) + 1;

  if (_cursor_out.at(r) >= _boundary_out.at(r))
  {
    if (r > 0)
    {
      _cursor_out.at(r) = 0;
      increment(r - 1);
    }
    else
    {
      // reached to the end
    }
  }
}

void TensorPackEnumerator::start(void)
{
  uint32_t rank = _rank_out;

  _cursor_out.resize(rank);
  _boundary_out.resize(rank);
  for (uint32_t r = 0; r < rank; ++r)
  {
    _cursor_out.at(r) = 0;
    _boundary_out.at(r) = _shape_out.dim(r).value();
  }

  rank = _rank_out - 1;
  _cursor_inp.resize(rank);
  _boundary_inp.resize(rank);
  for (uint32_t r = 0; r < rank; ++r)
  {
    _cursor_inp.at(r) = 0;
    _boundary_inp.at(r) = _shape_inp.dim(r).value();
  }
  _num_inp = 0;
}

bool TensorPackEnumerator::valid(void)
{
  uint32_t rank = _rank_out;
  for (uint32_t r = 0; r < rank; ++r)
  {
    if (_cursor_out.at(r) >= _boundary_out.at(r))
    {
      return false;
    }
  }
  return true;
}

void TensorPackEnumerator::advance(void)
{
  uint32_t r = _rank_out - 1;
  increment(r);

  // from _cursor_out, set _cursor_inp and _num
  for (int32_t r = 0, s = 0; r < _rank_out; ++r)
  {
    if (r == _axis)
    {
      _num_inp = _cursor_out.at(r);
    }
    else
    {
      _cursor_inp.at(s) = _cursor_out.at(r);
      s++;
    }
  }
}

uint32_t TensorPackEnumerator::inp_num(void) const { return _num_inp; }

uint32_t TensorPackEnumerator::inp_element(void) const
{
  uint32_t rank = _rank_out - 1;
  uint32_t element = 0;
  for (uint32_t r = 0; r < rank; ++r)
  {
    uint32_t dim = _boundary_inp.at(r);
    element = element * dim + _cursor_inp.at(r);
  }
  return element;
}

uint32_t TensorPackEnumerator::out_element(void) const
{
  uint32_t rank = _rank_out;
  uint32_t element = 0;
  for (uint32_t r = 0; r < rank; ++r)
  {
    uint32_t dim = _boundary_out.at(r);
    element = element * dim + _cursor_out.at(r);
  }
  return element;
}

} // namespace moco
