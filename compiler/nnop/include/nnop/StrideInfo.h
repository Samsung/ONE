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

#ifndef __NNOP_STRIDE_INFO_H__
#define __NNOP_STRIDE_INFO_H__

#include <cstdint>

namespace nnop
{

class StrideInfo
{
public:
  StrideInfo(uint32_t vertical, uint32_t horizontal) : _vertical{vertical}, _horizontal{horizontal}
  {
    // DO NOTHING
  }

public:
  uint32_t vertical(void) const { return _vertical; }
  uint32_t horizontal(void) const { return _horizontal; }

private:
  uint32_t _horizontal;
  uint32_t _vertical;
};

} // namespace nnop

#endif // __NNOP_STRIDE_INFO_H__
