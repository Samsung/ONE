/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LayoutSet.h"

namespace onert
{
namespace ir
{

LayoutSet::LayoutSet(std::initializer_list<Layout> layouts)
{
  for (auto &&layout : layouts)
  {
    _set.insert(layout);
  }
}

LayoutSet LayoutSet::operator|(const LayoutSet &other) const
{
  auto ret = *this;
  for (auto &&layout : other)
  {
    ret.add(layout);
  }
  return ret;
}

LayoutSet LayoutSet::operator&(const LayoutSet &other) const
{
  LayoutSet ret;
  for (auto &&layout : other)
  {
    if (contains(layout))
    {
      ret.add(layout);
    }
  }
  return ret;
}

LayoutSet LayoutSet::operator-(const LayoutSet &other) const
{
  auto ret = *this;
  for (auto &&layout : other)
  {
    ret.remove(layout);
  }
  return ret;
}

} // namespace ir
} // namespace onert
