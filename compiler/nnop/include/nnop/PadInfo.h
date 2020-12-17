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

#ifndef __NNOP_PAD_INFO_H__
#define __NNOP_PAD_INFO_H__

#include <cstdint>

namespace nnop
{

class PadInfo
{
public:
  PadInfo(uint32_t top, uint32_t bottom, uint32_t left, uint32_t right)
    : _top{top}, _bottom{bottom}, _left{left}, _right{right}
  {
    // DO NOTHING
  }

public:
  uint32_t top(void) const { return _top; }
  uint32_t bottom(void) const { return _bottom; }
  uint32_t left(void) const { return _left; }
  uint32_t right(void) const { return _right; }

private:
  uint32_t _top;
  uint32_t _bottom;
  uint32_t _left;
  uint32_t _right;
};

} // namespace nnop

#endif // __NNCC_CORE_PAD_INFO_H__
