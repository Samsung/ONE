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

#ifndef __ONERT_MODEL_OPERAND_INDEX_SEQUENCE_H__
#define __ONERT_MODEL_OPERAND_INDEX_SEQUENCE_H__

#include <initializer_list>
#include <vector>

#include "ir/Index.h"

namespace onert
{
namespace ir
{

class OperandIndexSequence
{
public:
  OperandIndexSequence(void) = default;
  OperandIndexSequence(std::initializer_list<OperandIndex> list);
  OperandIndexSequence(std::initializer_list<int32_t> list);
  OperandIndexSequence(std::initializer_list<uint32_t> list);

public:
  void append(const OperandIndex &index) { _vec.emplace_back(index); }
  void append(const OperandIndexSequence &l) { _vec.insert(_vec.end(), l.begin(), l.end()); }

public:
  uint32_t size() const { return static_cast<uint32_t>(_vec.size()); }
  const OperandIndex &at(IOIndex set_index) const { return _vec.at(set_index.value()); }
  const OperandIndex &at(uint32_t index) const { return _vec.at(index); }
  bool contains(const OperandIndex &index) const;
  void replace(const OperandIndex &from, const OperandIndex &to);
  OperandIndexSequence asUnique(void) const
  {
    ir::OperandIndexSequence unique;
    for (const auto &ind : _vec)
    {
      if (!unique.contains(ind))
      {
        unique.append(ind);
      }
    }
    return unique;
  }

public:
  OperandIndexSequence operator+(const OperandIndexSequence &other) const;

public:
  std::vector<OperandIndex>::const_iterator begin(void) const { return _vec.begin(); }
  std::vector<OperandIndex>::const_iterator end(void) const { return _vec.end(); }

private:
  std::vector<OperandIndex> _vec;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_MODEL_OPERAND_INDEX_SET_H__
