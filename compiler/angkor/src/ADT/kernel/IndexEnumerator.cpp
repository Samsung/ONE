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

#include "nncc/core/ADT/kernel/IndexEnumerator.h"

#include <cassert>
#include <algorithm>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace kernel
{

IndexEnumerator::IndexEnumerator(const Shape &shape) : _cursor(0)
{
  _max[0] = shape.width();
  _max[1] = shape.height();
  _max[2] = shape.depth();
  _max[3] = shape.count();

  std::fill(_cur, _cur + 4, 0);

  // NOTE Null dimension should NOT exist
  assert(std::find(_max, _max + 4, 0) == (_max + 4));
}

bool IndexEnumerator::valid(void) const { return _cursor < 4; }

uint32_t IndexEnumerator::count(void) const { return _cur[3]; }
uint32_t IndexEnumerator::depth(void) const { return _cur[2]; }
uint32_t IndexEnumerator::height(void) const { return _cur[1]; }
uint32_t IndexEnumerator::width(void) const { return _cur[0]; }

void IndexEnumerator::advance(void)
{
  while (_cursor < 4)
  {
    if (_cur[_cursor] + 1 < _max[_cursor])
    {
      break;
    }

    ++_cursor;
  }

  if (_cursor == 4)
  {
    return;
  }

  // Increment index
  _cur[_cursor] += 1;

  // Reset indices for lower dimensions
  for (uint32_t head = 0; head < _cursor; ++head)
  {
    _cur[head] = 0;
  }

  // Reset cursor
  _cursor = 0;
}

} // namespace kernel
} // namespace ADT
} // namespace core
} // namespace nncc
