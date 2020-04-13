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

#include "MemoryContext.h"

#include <cassert>

namespace enco
{

bool MemoryContext::member(const coco::Bag *bag) const
{
  // NOTE _base and _size SHOULD BE consistent
  if (_base.find(bag) != _base.end())
  {
    assert(_size.find(bag) != _size.end());
    return true;
  }

  assert(_size.find(bag) == _size.end());
  return false;
}

void MemoryContext::base(const coco::Bag *bag, const std::string &exp) { _base[bag] = exp; }
void MemoryContext::size(const coco::Bag *bag, const std::string &exp) { _size[bag] = exp; }

} // namespace enco
