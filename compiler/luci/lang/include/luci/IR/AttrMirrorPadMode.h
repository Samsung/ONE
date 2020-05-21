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

#ifndef __LUCI_IR_ATTR_MIRROR_PAD_MODE_H__
#define __LUCI_IR_ATTR_MIRROR_PAD_MODE_H__

namespace luci
{

enum class MirrorPadMode
{
  UNDEFINED, // This is not defined by Circle. This was added to prevent programming error.

  REFLECT,
  SYMMETRIC,
};

} // namespace luci

#endif // __LUCI_IR_ATTR_MIRROR_PAD_MODE_H__
