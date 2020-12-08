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

#include "luci/IR/AttrDilation.h"

#include <cassert>

namespace luci
{

void Dilation::w(int32_t w)
{
  assert(w >= 0);
  _w = static_cast<uint32_t>(w);
}

void Dilation::h(int32_t h)
{
  assert(h >= 0);
  _h = static_cast<uint32_t>(h);
}

} // namespace luci
