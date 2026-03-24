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

#include "coco/IR/ObjectManager.h"

#include "coco/IR/FeatureObject.h"
#include "coco/IR/KernelObject.h"

#include <memory>
#include <cassert>

using std::make_unique;

namespace coco
{

template <> FeatureObject *ObjectManager::create(void)
{
  auto feature = make_unique<FeatureObject>();
  modulize(feature.get());
  return take(std::move(feature));
}

template <> KernelObject *ObjectManager::create(void)
{
  auto kernel = make_unique<KernelObject>();
  modulize(kernel.get());
  return take(std::move(kernel));
}

void ObjectManager::destroy(Object *o)
{
  assert(o->def() == nullptr);
  assert(o->uses()->size() == 0);
  release(o);
}

} // namespace coco
