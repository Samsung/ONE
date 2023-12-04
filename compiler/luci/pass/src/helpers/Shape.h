/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PASS_HELPERS_SHAPE_H__
#define __LUCI_PASS_HELPERS_SHAPE_H__

#include <luci/IR/CircleNodes.h>

namespace luci
{

bool is_same_shape(const luci::CircleNode *node, const loco::TensorShape &shape);
bool is_same_shape(const luci::CircleNode *node, const std::initializer_list<uint32_t> shape);

bool has_dynamic_shape(const loco::Node *node);

} // namespace luci

#endif // __LUCI_PASS_HELPERS_SHAPE_H__
