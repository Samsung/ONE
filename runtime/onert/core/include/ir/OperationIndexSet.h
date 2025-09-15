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

#ifndef __ONERT_MODEL_OPERATION_INDEX_SET_H__
#define __ONERT_MODEL_OPERATION_INDEX_SET_H__

#include "ir/Index.h"

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <unordered_set>

namespace onert::ir
{

class OperationIndexSet
{
public:
  OperationIndexSet(void) = default;
  OperationIndexSet(std::initializer_list<OperationIndex> list);

public:
  void insert(const OperationIndex &index) { _set.insert(index); }
  void clear(void) { _set.clear(); }
  void remove(const OperationIndex &index)
  {
    auto itr = std::find(_set.begin(), _set.end(), index);
    assert(itr != _set.end());
    _set.erase(itr);
  }

public:
  uint32_t size() const { return static_cast<uint32_t>(_set.size()); }
  bool contains(const OperationIndex &index) const;

public:
  std::unordered_set<OperationIndex>::iterator begin(void) { return _set.begin(); }
  std::unordered_set<OperationIndex>::iterator end(void) { return _set.end(); }
  std::unordered_set<OperationIndex>::const_iterator begin(void) const { return _set.begin(); }
  std::unordered_set<OperationIndex>::const_iterator end(void) const { return _set.end(); }

private:
  std::unordered_set<OperationIndex> _set;
};

} // namespace onert::ir

#endif // __ONERT_MODEL_OPERATION_INDEX_SET_H__
