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

#ifndef __LUCI_IR_CIRCLE_TOPK_V2_OUT_H__
#define __LUCI_IR_CIRCLE_TOPK_V2_OUT_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief Virtual CIRCLETOPKV2OUT in Circle
 */
class CircleTopKV2Out final
  : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::CIRCLETOPKV2OUT>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

public:
  int32_t index(void) const { return _index; }
  void index(int32_t index) { _index = index; }

private:
  int32_t _index{-1};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_TOPK_V2_OUT_H__
