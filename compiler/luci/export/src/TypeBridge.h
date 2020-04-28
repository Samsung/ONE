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

#ifndef __TYPE_BRIDGE_H__
#define __TYPE_BRIDGE_H__

#include <luci/IR/CircleNode.h>

#include <loco.h>

namespace luci
{

/**
 * @brief  node_shape() will return loco::TensorShape of CircleNode
 */
loco::TensorShape node_shape(CircleNode *node);

/**
 * @brief  node_dtype() will return loco::DataType of CircleNode
 */
loco::DataType node_dtype(CircleNode *node);

/**
 * @brief copy_shape_dtype() will copy shape and dtype inference data to CircleNode
 */
void copy_shape_dtype(loco::Graph *graph);

} // namespace luci

#endif // __TYPE_BRIDGE_H__
