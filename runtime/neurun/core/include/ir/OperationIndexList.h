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

#ifndef __NEURUN_MODEL_OPERATION_INDEX_LIST_H__
#define __NEURUN_MODEL_OPERATION_INDEX_LIST_H__

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <list>

#include "ir/Index.h"

namespace neurun
{
namespace ir
{

class OperationIndexList
{
public:
  OperationIndexList(void) = default;
  OperationIndexList(std::initializer_list<OperationIndex> list);

public:
  void append(const OperationIndex &index) { _list.push_back(index); }
  void remove(const OperationIndex &index)
  {
    auto itr = std::find(_list.begin(), _list.end(), index);
    assert(itr != _list.end());
    _list.erase(itr);
  }

public:
  uint32_t size() const { return static_cast<uint32_t>(_list.size()); }
  const std::list<OperationIndex> &list() const { return _list; }
  bool contains(const OperationIndex &index) const;

private:
  std::list<OperationIndex> _list;
};

} // namespace ir
} // namespace neurun

#endif // __NEURUN_MODEL_OPERATION_INDEX_LIST_H__
