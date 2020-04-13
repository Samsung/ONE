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

#ifndef __COCO_IR_ELEM_ID_H__
#define __COCO_IR_ELEM_ID_H__

#include <cstdint>

namespace coco
{

class ElemID final
{
public:
  ElemID() : _value{0xffffffff}
  {
    // DO NOTHING
  }

public:
  explicit ElemID(uint32_t value) : _value{value}
  {
    // DO NOTHING
  }

public:
  uint32_t value(void) const { return _value; }

private:
  uint32_t _value;
};

bool operator==(const ElemID &lhs, const ElemID &rhs);
bool operator<(const ElemID &lhs, const ElemID &rhs);

} // namespace coco

#endif // __COCO_IR_ELEM_ID_H__
