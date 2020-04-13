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

#ifndef __COCO_IR_OBJECT_MANAGER_H__
#define __COCO_IR_OBJECT_MANAGER_H__

#include "coco/IR/Object.h"
#include "coco/IR/FeatureShape.h"
#include "coco/IR/FeatureObject.h"
#include "coco/IR/KernelObject.forward.h"
#include "coco/IR/EntityBuilder.h"

#include "coco/ADT/PtrManager.h"

#include <nncc/core/ADT/kernel/Shape.h>

namespace coco
{

class ObjectManager final : public PtrManager<Object>, public EntityBuilder
{
public:
  ObjectManager(Module *m = nullptr) { module(m); }

public:
  template <typename T> T *create(void);

public:
  /**
   * @brief Destroy (= deallocate) an Object entity
   *
   * NOTE An Object SHOULD HAVE NO DEF & USES to be destructed
   * NOTE An Object WILL BE unlinked from its dependent bag (if has) on destruction
   */
  void destroy(Object *o);
};

} // namespace coco

#endif // __COCO_IR_OBJECT_MANAGER_H__
