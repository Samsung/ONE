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

#ifndef __LUCI_IR_CIRCLE_SPLIT_H__
#define __LUCI_IR_CIRCLE_SPLIT_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief SPLIT in Circle
 */
class CircleSplit final : public FixedArityNode<2, CircleNodeImpl<CircleOpcode::SPLIT>>
{
public:
  loco::Node *split_dim(void) const { return at(0)->node(); }
  void split_dim(loco::Node *node) { at(0)->node(node); }

  loco::Node *input(void) const { return at(1)->node(); }
  void input(loco::Node *node) { at(1)->node(node); }

public:
  // NOTE it is num_split() not num_splits() as we follow TF name
  int32_t num_split(void) const { return _num_split; }
  void num_split(int32_t num_split) { _num_split = num_split; }

private:
  int32_t _num_split{0};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_SPLIT_H__
