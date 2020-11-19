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

#ifndef __LUCI_CIRCLE_REMOVE_DUPLICATE_TRANSPOSE_PASS_H__
#define __LUCI_CIRCLE_REMOVE_DUPLICATE_TRANSPOSE_PASS_H__

#include <luci/IR/CircleNodes.h>

namespace luci
{

/// @return true if Combination of two Const permutation is [0, 1, 2, ...., n]'
bool check_perm(const luci::CircleConst *pred_perm, const luci::CircleConst *main_perm);

/// @return true if Duplicated Transpose is fused or removed'
bool remove_duplicate_transpose_function(luci::CircleNode *node);

} // namespace luci

#endif // __LUCI_CIRCLE_REMOVE_DUPLICATE_TRANSPOSE_PASS_H__
