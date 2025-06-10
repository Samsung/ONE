/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ArrayIndex.h"

#include <cassert>
#include <stdexcept>

namespace luci
{

#define THROW_UNLESS(COND) \
  if (not(COND))           \
    throw std::invalid_argument("");

Array4DIndex::Array4DIndex(uint32_t D0, uint32_t D1, uint32_t D2, uint32_t D3)
  : _dim{D0, D1, D2, D3}
{
  _strides[3] = 1;
  _strides[2] = D3;
  _strides[1] = D3 * D2;
  _strides[0] = D3 * D2 * D1;

  for (int i = 0; i < 4; ++i)
  {
    THROW_UNLESS(_strides[i] > 0);
  }
}

uint32_t Array4DIndex::operator()(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) const
{
  THROW_UNLESS(i0 < _dim[0] && i1 < _dim[1] && i2 < _dim[2] && i3 < _dim[3]);

  return i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2] + i3 * _strides[3];
}

uint32_t Array4DIndex::size(void) const
{

  for (int i = 0; i < 4; ++i)
  {
    THROW_UNLESS(_dim[i] > 0);
  }

  return _dim[0] * _dim[1] * _dim[2] * _dim[3];
}

uint32_t Array4DIndex::stride(uint32_t axis) const { return _strides[axis]; }

} // namespace luci
