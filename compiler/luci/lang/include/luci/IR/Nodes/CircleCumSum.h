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

#ifndef __LUCI_IR_CIRCLE_CUMSUM_H__
#define __LUCI_IR_CIRCLE_CUMSUM_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleNodeMixins.h"
#include "luci/IR/CircleOpcode.h"

namespace luci
{

class CircleCumSum final : public FixedArityNode<2, CircleNodeImpl<CircleOpcode::CUMSUM>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *axis(void) const { return at(1)->node(); }
  void axis(loco::Node *node) { at(1)->node(node); }

public:
  bool exclusive(void) const { return _exclusive; }
  void exclusive(bool exclusive) { _exclusive = exclusive; }

  bool reverse(void) const { return _reverse; }
  void reverse(bool reverse) { _reverse = reverse; }

private:
  bool _exclusive{false};
  bool _reverse{false};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_CUMSUM_H__
