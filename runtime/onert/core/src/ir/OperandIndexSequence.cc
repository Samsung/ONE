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

#include "ir/OperandIndexSequence.h"

#include <algorithm>

namespace onert
{
namespace ir
{

OperandIndexSequence::OperandIndexSequence(std::initializer_list<OperandIndex> list) : _set(list)
{
  // DO NOTHING
}

OperandIndexSequence::OperandIndexSequence(std::initializer_list<int32_t> list)
{
  for (auto val : list)
  {
    _set.emplace_back(static_cast<uint32_t>(val));
  }
}

OperandIndexSequence::OperandIndexSequence(std::initializer_list<uint32_t> list)
{
  for (auto val : list)
  {
    _set.emplace_back(val);
  }
}

bool OperandIndexSequence::contains(const OperandIndex &index) const
{
  return std::find(_set.begin(), _set.end(), index) != _set.end();
}

void OperandIndexSequence::replace(const OperandIndex &from, const OperandIndex &to)
{
  std::replace(_set.begin(), _set.end(), from, to);
}

OperandIndexSequence OperandIndexSequence::operator+(const OperandIndexSequence &other) const
{
  OperandIndexSequence ret = *this;
  ret.append(other);
  return ret;
}

} // namespace ir
} // namespace onert
