/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_MODEL_OPERAND_INDEX_SEQUENCE_H__
#define __ONERT_IR_MODEL_OPERAND_INDEX_SEQUENCE_H__

#include "ir/Index.h"

#include <initializer_list>
#include <vector>

namespace onert
{
namespace ir
{

class ModelOperandIndexSequence
{
public:
  ModelOperandIndexSequence(void) = default;
  ModelOperandIndexSequence(std::initializer_list<ModelOperandIndex> list);
  ModelOperandIndexSequence(std::initializer_list<int32_t> list);
  ModelOperandIndexSequence(std::initializer_list<uint32_t> list);

public:
  void append(const ModelOperandIndex &index) { _vec.emplace_back(index); }
  void append(const ModelOperandIndexSequence &l) { _vec.insert(_vec.end(), l.begin(), l.end()); }

public:
  uint32_t size() const { return static_cast<uint32_t>(_vec.size()); }
  const ModelOperandIndex &at(IOIndex set_index) const { return _vec.at(set_index.value()); }
  const ModelOperandIndex &at(uint32_t index) const { return _vec.at(index); }
  bool contains(const ModelOperandIndex &index) const;
  void replace(const ModelOperandIndex &from, const ModelOperandIndex &to);

public:
  ModelOperandIndexSequence operator+(const ModelOperandIndexSequence &other) const;

public:
  std::vector<ModelOperandIndex>::const_iterator begin(void) const { return _vec.begin(); }
  std::vector<ModelOperandIndex>::const_iterator end(void) const { return _vec.end(); }
  std::vector<ModelOperandIndex>::iterator begin(void) { return _vec.begin(); }
  std::vector<ModelOperandIndex>::iterator end(void) { return _vec.end(); }

private:
  std::vector<ModelOperandIndex> _vec;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_MODEL_OPERAND_INDEX_SEQUENCE_H__
