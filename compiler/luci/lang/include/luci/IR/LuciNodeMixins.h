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

#ifndef __LUCI_IR_LUCINODEMIXINS_H__
#define __LUCI_IR_LUCINODEMIXINS_H__

// TODO remove this file after LuciNodeTrait and LuciNodeMixin are not used in backend

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

using LuciNodeTrait = CircleNodeTrait;

template <LuciNodeTrait T> using LuciNodeMixin = CircleNodeMixin<T>;

} // namespace luci

#endif // __LUCI_IR_LUCINODEMIXINS_H__
