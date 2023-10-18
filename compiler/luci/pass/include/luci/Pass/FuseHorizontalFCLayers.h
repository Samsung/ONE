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

#ifndef __LUCI_FUSE_HORIZONTAL_FULLY_CONNECTED_PASS_H__
#define __LUCI_FUSE_HORIZONTAL_FULLY_CONNECTED_PASS_H__

#include <logo/Pass.h>

namespace luci
{

/**
 * @brief  Class to fuse horizontal FC layers
 *
 *  Before
 *
 *     +---- [In] ----+
 *     |              |
 *     V              V
 *   fc1 (w1, b1)   fc2 (w2, b2)
 *     |              |
 *     |              |
 *     +---> add <----+
 *            |
 *            V
 *          [Out]
 *
 *  After
 *
 *     [In]
 *      |
 *      V
 *     fc3 (w1+w2, b1+b2)
 *      |
 *      V
 *     [Out]
 *
 *     Shape/dtype of fc1, fc2, and fc3 should be the same.
 */
struct FuseHorizontalFullyConnectedPass final : public logo::Pass
{
  const char *name(void) const final { return "luci::FuseHorizontalFullyConnectedPass"; }

  bool run(loco::Graph *g) final;
};

} // namespace luci

#endif // __LUCI_FUSE_HORIZONTAL_FULLY_CONNECTED_PASS_H__
