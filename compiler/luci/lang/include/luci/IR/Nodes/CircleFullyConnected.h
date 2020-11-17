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

#ifndef __LUCI_IR_CIRCLEFULLYCONNECTED_H__
#define __LUCI_IR_CIRCLEFULLYCONNECTED_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief FULLY_CONNECTED in Circle
 */
class CircleFullyConnected final
    : public FixedArityNode<3, CircleNodeImpl<CircleOpcode::FULLY_CONNECTED>>,
      public LuciNodeMixin<LuciNodeTrait::FusedActFunc>,
      public LuciNodeMixin<LuciNodeTrait::Bias>
{
public:
  enum class WeightsFormat
  {
    UNDEFINED, // This is not defined by Circle. This was added to prevent programming error.

    DEFAULT,
    SHUFFLED4x16INT8,
    SHUFFLED16x1FLOAT32,
  };

public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *weights(void) const { return at(1)->node(); }
  void weights(loco::Node *node) { at(1)->node(node); }

  loco::Node *bias(void) const override { return at(2)->node(); }
  void bias(loco::Node *node) override { at(2)->node(node); }

public:
  WeightsFormat weights_format(void) const { return _weights_format; }
  void weights_format(WeightsFormat weights_format) { _weights_format = weights_format; }

private:
  WeightsFormat _weights_format{WeightsFormat::DEFAULT};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEFULLYCONNECTED_H__
