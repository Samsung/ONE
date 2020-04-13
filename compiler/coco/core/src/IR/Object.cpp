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

#include "coco/IR/Object.h"
#include "coco/IR/Def.h"
#include "coco/IR/Use.h"

#include <cassert>
#include <stdexcept>

namespace coco
{

Object::Object()
{
  // Register self to Dep
  _dep.object(this);
}

Def *Object::def(void) const { return _def; }

void Object::def(Def *d)
{
  // This assert enforces users to explicitly reset def before update.
  //
  // Let's consider an object o with def d0.
  //
  // The following code is allowed:
  // o->def(nullptr);
  // o->def(d1);
  //
  // However, the following code is not allowed:
  // o->def(d1);
  //
  assert((_def == nullptr) || (d == nullptr));
  _def = d;
}

const UseSet *Object::uses(void) const { return &_uses; }
UseSet *Object::mutable_uses(void) { return &_uses; }

Object::Producer *producer(const Object *obj)
{
  if (auto d = obj->def())
  {
    return d->producer();
  }

  return nullptr;
}

Object::ConsumerSet consumers(const Object *obj)
{
  Object::ConsumerSet res;

  for (const auto &use : *(obj->uses()))
  {
    if (auto consumer = use->consumer())
    {
      res.insert(consumer);
    }
  }

  return res;
}

/**
 * Casting Helpers
 *
 * TODO Use Macro to reduce code duplication
 */
template <> bool isa<FeatureObject>(const Object *o) { return o->asFeature() != nullptr; }
template <> bool isa<KernelObject>(const Object *o) { return o->asKernel() != nullptr; }

template <> FeatureObject *cast(Object *o)
{
  assert(o != nullptr);
  auto res = o->asFeature();
  assert(res != nullptr);
  return res;
}

template <> KernelObject *cast(Object *o)
{
  assert(o != nullptr);
  auto res = o->asKernel();
  assert(res != nullptr);
  return res;
}

template <> FeatureObject *safe_cast(Object *o)
{
  // NOTE o may be nullptr
  return (o == nullptr) ? nullptr : o->asFeature();
}

template <> KernelObject *safe_cast(Object *o)
{
  // NOTE o may be nullptr
  return (o == nullptr) ? nullptr : o->asKernel();
}

} // namespace coco
