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

#ifndef __COCO_IR_UPDATE_H__
#define __COCO_IR_UPDATE_H__

#include "coco/IR/Bag.h"

namespace coco
{

/**
 * @brief A Update represents an edge between a Bag and its Updater
 */
class Update final
{
public:
  Update(Bag::Updater *u) { updater(u); }

public:
  ~Update();

public:
  Bag *bag(void) const { return _bag; }
  void bag(Bag *bag);

public:
  Bag::Updater *updater(void) const { return _updater; }
  void updater(Bag::Updater *u) { _updater = u; }

private:
  Bag *_bag = nullptr;
  Bag::Updater *_updater = nullptr;
};

} // namespace coco

#endif // __COCO_IR_UPDATE_H__
