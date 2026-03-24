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

#ifndef __COCO_IR_READ_H__
#define __COCO_IR_READ_H__

#include "coco/IR/Bag.h"

namespace coco
{

/**
 * @brief A Read represents an edge between a Bag and its Reader
 */
class Read final
{
public:
  Read(Bag::Reader *r)
  {
    // Initialize link and reader
    reader(r);
  }

public:
  ~Read();

public:
  Bag *bag(void) const { return _bag; }
  void bag(Bag *bag);

public:
  Bag::Reader *reader(void) const { return _reader; }
  void reader(Bag::Reader *r) { _reader = r; }

private:
  Bag *_bag = nullptr;
  Bag::Reader *_reader = nullptr;
};

} // namespace coco

#endif // __COCO_IR_READ_H__
