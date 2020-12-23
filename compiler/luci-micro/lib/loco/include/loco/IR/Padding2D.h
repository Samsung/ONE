/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_PADDING2D_H__
#define __LOCO_IR_PADDING2D_H__

#include <cstdint>

namespace loco
{

class Padding2D final
{
public:
  Padding2D() : _top{0}, _bottom{0}, _left{0}, _right{0}
  {
    // DO NOTHING
  }

public:
  Padding2D(uint32_t top, uint32_t bottom, uint32_t left, uint32_t right)
      : _top{top}, _bottom{bottom}, _left{left}, _right{right}
  {
    // DO NOTHING
  }

public:
  uint32_t top(void) const { return _top; }
  void top(uint32_t value) { _top = value; }

public:
  uint32_t bottom(void) const { return _bottom; }
  void bottom(uint32_t value) { _bottom = value; }

public:
  uint32_t left(void) const { return _left; }
  void left(uint32_t value) { _left = value; }

public:
  uint32_t right(void) const { return _right; }
  void right(uint32_t value) { _right = value; }

private:
  uint32_t _top;
  uint32_t _bottom;
  uint32_t _left;
  uint32_t _right;
};

} // namespace loco

#endif // __LOCO_IR_PADDING2D_H__
