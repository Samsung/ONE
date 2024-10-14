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

#ifndef __LUCI_IR_CIRCLERMSNORM_H__
#define __LUCI_IR_CIRCLERMSNORM_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief RMS_NORM in Circle
 */
class CircleRmsNorm final : public FixedArityNode<2, CircleNodeImpl<CircleOpcode::RMS_NORM>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *gamma(void) const { return at(1)->node(); }
  void gamma(loco::Node *node) { at(1)->node(node); }

public:
  float epsilon() const { return _epsilon; }
  void epsilon(float epsilon) { _epsilon = epsilon; }

private:
  float _epsilon{1e-06};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLERMSNORM_H__
