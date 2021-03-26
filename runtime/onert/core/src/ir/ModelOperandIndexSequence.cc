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

#include "ir/ModelOperandIndexSequence.h"

#include <algorithm>

namespace onert
{
namespace ir
{

ModelOperandIndexSequence::ModelOperandIndexSequence(std::initializer_list<ModelOperandIndex> list)
  : _vec(list)
{
  // DO NOTHING
}

ModelOperandIndexSequence::ModelOperandIndexSequence(std::initializer_list<int32_t> list)
{
  for (auto val : list)
  {
    _vec.emplace_back(static_cast<uint32_t>(val));
  }
}

ModelOperandIndexSequence::ModelOperandIndexSequence(std::initializer_list<uint32_t> list)
{
  for (auto val : list)
  {
    _vec.emplace_back(val);
  }
}

bool ModelOperandIndexSequence::contains(const ModelOperandIndex &index) const
{
  return std::find(_vec.begin(), _vec.end(), index) != _vec.end();
}

void ModelOperandIndexSequence::replace(const ModelOperandIndex &from, const ModelOperandIndex &to)
{
  std::replace(_vec.begin(), _vec.end(), from, to);
}

ModelOperandIndexSequence ModelOperandIndexSequence::
operator+(const ModelOperandIndexSequence &other) const
{
  ModelOperandIndexSequence ret = *this;
  ret.append(other);
  return ret;
}

} // namespace ir
} // namespace onert
