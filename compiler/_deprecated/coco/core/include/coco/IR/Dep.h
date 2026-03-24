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

#ifndef __COCO_IR_DEP_H__
#define __COCO_IR_DEP_H__

#include "coco/IR/Bag.h"
#include "coco/IR/Object.forward.h"

namespace coco
{

/**
 * @brief A Dep represents the edge between a Bag and its dependent Object
 *
 * WARNING A Dep will update dependent Object set (stored BagInfo) only when
 * users properly initialize object and link values.
 */
class Dep final
{
public:
  Dep() = default;

public:
  Dep(const Dep &) = delete;
  Dep(Dep &&) = delete;

public:
  ~Dep();

public:
  Bag *bag(void) const { return _bag; }
  void bag(Bag *);

public:
  Object *object(void) const { return _object; }
  void object(Object *object) { _object = object; }

private:
  Bag *_bag = nullptr;
  Object *_object = nullptr;
};

} // namespace coco

#endif // __COCO_IR_DEP_H__
