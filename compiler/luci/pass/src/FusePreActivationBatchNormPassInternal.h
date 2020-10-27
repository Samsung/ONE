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

#ifndef __LUCI_CIRCLE_FUSE_PRE_ACTIVATION_BATCH_NORM_PASS_INTERNAL_H__
#define __LUCI_CIRCLE_FUSE_PRE_ACTIVATION_BATCH_NORM_PASS_INTERNAL_H__

#include <luci/IR/CircleNodes.h>

namespace luci
{

//  Swap MUL/ADD if they are from batch normalization
/// @return true if success
bool swap_mul_add(luci::CircleAdd *add, std::vector<luci::CircleMul *> &mul_list,
                  std::vector<luci::CircleAdd *> &add_list);

//  Fuse MUL with the next CONV if possible
/// @return true if success
bool fuse_mul_with_conv(luci::CircleMul *mul);

//  Fuse ADD with the preceding CONV if possible
/// @return true if success
bool fuse_add_with_conv(luci::CircleAdd *mul, std::vector<luci::CircleSub *> &sub_list);

//  Fuse SUB with CONV if possible
/// @return true if success
bool fuse_sub_with_conv(luci::CircleSub *sub);

} // namespace luci

#endif // __LUCI_CIRCLE_FUSE_PRE_ACTIVATION_BATCH_NORM_PASS_INTERNAL_H__
