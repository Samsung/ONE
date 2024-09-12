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

#ifndef __LUCI_IR_CIRCLEROPE_H__
#define __LUCI_IR_CIRCLEROPE_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"
#include "luci/IR/AttrRoPEMode.h"

namespace luci
{

/**
 * @brief ROPE in Circle
 */
class CircleRoPE final : public FixedArityNode<3, CircleNodeImpl<CircleOpcode::ROPE>>
{
public:
  /// @note  Currently only support FLOAT32 as input node
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *sin_table(void) const { return at(1)->node(); }
  void sin_table(loco::Node *node) { at(1)->node(node); }

  loco::Node *cos_table(void) const { return at(2)->node(); }
  void cos_table(loco::Node *node) { at(2)->node(node); }

public:
  RoPEMode mode() const { return _mode; }
  void mode(RoPEMode mode) { _mode = mode; }

private:
  RoPEMode _mode{RoPEMode::GPT_NEOX};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEROPE_H__
