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

#include "coco/IR/Ops.h"

#include <pepper/assert.h>

namespace coco
{

Conv2D::Conv2D() : _ker{this}, _arg{this}
{
  // DO NOTHING
}

uint32_t Conv2D::arity(void) const
{
  // Conv2D has one argument (IFM)
  // NOTE This design is subject to change
  return 1;
}

Op *Conv2D::arg(DBGARG(uint32_t, n)) const
{
  assert(n < arity());
  return arg();
}

std::set<Object *> Conv2D::uses(void) const
{
  std::set<Object *> res;

  if (ker())
  {
    res.insert(ker());
  }

  if (auto ifm = arg())
  {
    for (auto obj : ifm->uses())
    {
      res.insert(obj);
    }
  }

  return res;
}

void Conv2D::ker(KernelObject *ker) { _ker.value(ker); }

KernelObject *Conv2D::ker(void) const
{
  if (auto obj = _ker.value())
  {
    assert(obj->asKernel() != nullptr);
    return obj->asKernel();
  }

  return nullptr;
}

} // namespace coco
