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

#include "coco/IR/Instrs.h"

namespace coco
{

uint32_t Shuffle::size(void) const { return _content.size(); }

std::set<ElemID> Shuffle::range(void) const
{
  std::set<ElemID> res;

  for (auto it = _content.begin(); it != _content.end(); ++it)
  {
    res.insert(it->first);
  }

  return res;
}

void Shuffle::insert(const ElemID &from, const ElemID &into) { _content[into] = from; }

void Shuffle::from(Bag *b) { _from.bag(b); }
void Shuffle::into(Bag *b) { _into.bag(b); }

} // namespace coco
