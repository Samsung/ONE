/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_ATTRSTRIDE_H__
#define __LUCI_IR_ATTRSTRIDE_H__

#include <stdint.h>

namespace luci
{

class Stride final
{
public:
  Stride() : _w(1), _h(1) {}

  uint32_t w() const { return _w; }
  void w(uint32_t w) { _w = w; }
  void w(int32_t w);

  uint32_t h() const { return _h; }
  void h(uint32_t h) { _h = h; }
  void h(int32_t h);

private:
  uint32_t _w;
  uint32_t _h;
};

} // namespace luci

#endif // __LUCI_IR_ATTRSTRIDE_H__
