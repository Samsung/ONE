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

#ifndef __LUCI_IR_CIRCLEINSTANCENORM_H__
#define __LUCI_IR_CIRCLEINSTANCENORM_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief INSTANCE_NORM in Circle
 */
class CircleInstanceNorm final
  : public FixedArityNode<3, CircleNodeImpl<CircleOpcode::INSTANCE_NORM>>,
    public CircleNodeMixin<CircleNodeTrait::FusedActFunc>
{
public:
  /// @note  Currently only support FLOAT32 as input node
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *gamma(void) const { return at(1)->node(); }
  void gamma(loco::Node *node) { at(1)->node(node); }

  loco::Node *beta(void) const { return at(2)->node(); }
  void beta(loco::Node *node) { at(2)->node(node); }

  float epsilon() const { return _epsilon; }
  void epsilon(float epsilon) { _epsilon = epsilon; }

private:
  float _epsilon = 1e-05;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEINSTANCENORM_H__
