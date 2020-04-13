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

#include "nest/Level.h"

#include <cassert>

namespace
{
const uint32_t invalid_tag = 0xffffffff;
} // namespace

namespace nest
{
Level::Level() : _value{invalid_tag}
{
  // DO NOTHING
}

Level::Level(uint32_t value) : _value{value} { assert(value != invalid_tag); }

bool Level::valid(void) const { return _value != invalid_tag; }

bool operator==(const Level &lhs, const Level &rhs)
{
  assert(lhs.valid() && rhs.valid());
  return lhs.value() == rhs.value();
}

bool operator<(const Level &lhs, const Level &rhs)
{
  assert(lhs.valid() && rhs.valid());
  return lhs.value() < rhs.value();
}
} // namespace nest
