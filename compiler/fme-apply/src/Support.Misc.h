/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __FME_APPLY_SUPPORT_MISC_H__
#define __FME_APPLY_SUPPORT_MISC_H__

#include <luci/IR/CircleNode.h>

namespace fme_apply
{

void copy_shape(luci::CircleNode *from, luci::CircleNode *to);
loco::Node *get_input(luci::CircleNode *node);
void set_input(luci::CircleNode *node, luci::CircleCustom *scale);
luci::CircleNode *find_arg_with_name(const luci::CircleNode *node, const std::string &name,
                                     const uint32_t &depth = 1);

} // namespace fme_apply

#endif //__FME_APPLY_SUPPORT_MISC_H__
