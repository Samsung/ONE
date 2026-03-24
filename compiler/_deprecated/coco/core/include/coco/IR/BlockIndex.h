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

#ifndef __COCO_IR_BLOCK_INDEX_H__
#define __COCO_IR_BLOCK_INDEX_H__

#include <cstdint>

namespace coco
{

/**
 * @brief A BlockIndex denotes the index of a block in a block list
 */
class BlockIndex final
{
private:
  static const uint32_t undefined = 0xffffffff;

public:
  BlockIndex() : _value{undefined}
  {
    // DO NOTHING
  }

public:
  BlockIndex(uint32_t value) { set(value); }

public:
  bool valid(void) const { return _value != undefined; }

public:
  uint32_t value(void) const { return _value; }

public:
  void set(uint32_t value);
  void reset(void) { _value = undefined; }

private:
  uint32_t _value;
};

static inline bool operator<(const BlockIndex &lhs, const BlockIndex &rhs)
{
  return lhs.value() < rhs.value();
}

} // namespace coco

#endif // __COCO_IR_BLOCK_INDEX_H__
