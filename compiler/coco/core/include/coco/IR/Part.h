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

#ifndef __COCO_IR_PART_H__
#define __COCO_IR_PART_H__

#include "coco/IR/Op.forward.h"

namespace coco
{

/**
 * @brief A Part represents the edge between a child Op and its parent Op
 */
class Part final
{
public:
  Part(Op *parent) : _parent{parent}
  {
    // DO NOTHING
  }

public:
  ~Part() { child(nullptr); }

public:
  Op *child(void) const { return _child; }
  void child(Op *c);

public:
  Op *parent(void) const { return _parent; }

private:
  Op *_parent = nullptr;
  Op *_child = nullptr;
};

} // namespace coco

#endif // __COCO_IR_PART_H__
