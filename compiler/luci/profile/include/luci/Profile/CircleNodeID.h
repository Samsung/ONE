/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PROFILE_CIRCLE_NODE_ID_H__
#define __LUCI_PROFILE_CIRCLE_NODE_ID_H__

#include <luci/IR/CircleNode.h>

namespace luci
{

using CircleNodeID = uint32_t;

bool has_node_id(luci::CircleNode *circle_node);

void set_node_id(luci::CircleNode *circle_node, CircleNodeID id);

CircleNodeID get_node_id(luci::CircleNode *circle_node);

} // namespace luci

#endif // __LUCI_PROFILE_CIRCLE_NODE_ID_H__
