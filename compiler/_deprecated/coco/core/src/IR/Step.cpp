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

#include "coco/IR/Step.h"
#include "coco/IR/Op.h"

#include <cassert>

namespace coco
{

void Step::op(Op *o)
{
  if (_op != nullptr)
  {
    // Unlink step from _op
    assert(_op->_step == this);
    _op->_step = nullptr;

    // Reset _op
    _op = nullptr;
  }

  assert(_op == nullptr);

  if (o)
  {
    // Update _op
    _op = o;

    // Link step to _op
    assert(_op->_step == nullptr);
    _op->_step = this;
  }

  assert(_op == o);
}

} // namespace coco
