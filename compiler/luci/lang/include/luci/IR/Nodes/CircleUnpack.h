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

#ifndef __LUCI_IR_CIRCLE_UNPACK_H__
#define __LUCI_IR_CIRCLE_UNPACK_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief UNPACK in Circle
 */
class CircleUnpack final : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::UNPACK>>
{
public:
  CircleUnpack() = default;

public:
  loco::Node *value(void) const { return at(0)->node(); }
  void value(loco::Node *node) { at(0)->node(node); }

public:
  int32_t num(void) const { return _num; }
  void num(int32_t num) { _num = num; }

  int32_t axis(void) const { return _axis; }
  void axis(int32_t axis) { _axis = axis; }

private:
  int32_t _num{0};
  int32_t _axis{0};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_UNPACK_H__
