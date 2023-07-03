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

#ifndef __ONERT_IR_LAYOUT_SET_H__
#define __ONERT_IR_LAYOUT_SET_H__

#include <cstdint>
#include <initializer_list>
#include <unordered_set>

#include "ir/Layout.h"

namespace onert
{
namespace ir
{

class LayoutSet
{
public:
  LayoutSet() = default;
  LayoutSet(std::initializer_list<Layout> layouts);

public:
  void add(const Layout &layout) { _set.insert(layout); }
  void remove(const Layout &layout) { _set.erase(layout); }
  uint32_t size() const { return static_cast<uint32_t>(_set.size()); }
  bool contains(const Layout &layout) const { return _set.find(layout) != _set.end(); }

public:
  LayoutSet operator|(const LayoutSet &other) const; // Union
  LayoutSet operator&(const LayoutSet &other) const; // Intersect
  LayoutSet operator-(const LayoutSet &other) const; // Minus

public:
  std::unordered_set<Layout>::const_iterator begin() const { return _set.begin(); }
  std::unordered_set<Layout>::const_iterator end() const { return _set.end(); }

private:
  std::unordered_set<Layout> _set;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_LAYOUT_SET_H__
