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

#include "coco/IR/Dep.h"
#include "coco/IR/Object.h"

#include <cassert>

namespace coco
{

Dep::~Dep() { bag(nullptr); }

void Dep::bag(Bag *bag)
{
  if (_bag != nullptr)
  {
    // Remove bag <-> dep link
    assert(_bag->deps()->find(this) != _bag->deps()->end());
    _bag->mutable_deps()->erase(this);

    // Reset _bag
    _bag = nullptr;
  }

  assert(_bag == nullptr);

  if (bag != nullptr)
  {
    // Set _bag
    _bag = bag;

    // Create bag <-> dep link
    _bag->mutable_deps()->insert(this);
  }

  assert(_bag == bag);
}

} // namespace coco
