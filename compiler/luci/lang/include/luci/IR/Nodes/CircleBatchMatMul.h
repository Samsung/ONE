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

#ifndef __LUCI_IR_CIRCELBATCHMATMUL_H__
#define __LUCI_IR_CIRCELBATCHMATMUL_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief BATCH_MATMUL in Circle
 */
class CircleBatchMatMul final : public FixedArityNode<2, CircleNodeImpl<CircleOpcode::BATCH_MATMUL>>
{
public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

  loco::Node *y(void) const { return at(1)->node(); }
  void y(loco::Node *node) { at(1)->node(node); }

public:
  bool adj_x(void) const { return _adj_x; }
  void adj_x(bool arg) { _adj_x = arg; }

  bool adj_y(void) const { return _adj_y; }
  void adj_y(bool arg) { _adj_y = arg; }

private:
  bool _adj_x = false;
  bool _adj_y = false;
};

} // namespace luci

#endif // __LUCI_IR_CIRCELBATCHMATMUL_H__
