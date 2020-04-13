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

#ifndef __COCO_IR_USE_H__
#define __COCO_IR_USE_H__

#include "coco/IR/Object.h"

namespace coco
{

class Use final
{
public:
  Use(Object::Consumer *use) : _value{nullptr}, _consumer{use}
  {
    // DO NOTHING
  }

public:
  ~Use() { value(nullptr); }

public:
  Object *value(void) const { return _value; }

public:
  void value(Object *value);

public:
  Object::Consumer *consumer(void) const { return _consumer; }

private:
  Object *_value;
  Object::Consumer *_consumer = nullptr;
};

} // namespace coco

#endif // __COCO_IR_USE_H__
