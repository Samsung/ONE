/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCOEX_IR_CIRCLENODES_H__
#define __LOCOEX_IR_CIRCLENODES_H__

#include "CircleNodeDecl.h"
#include "CircleOpcode.h"

#include "FusedActFunc.h"
#include "NodeMixins.h" // FixedArityNode

#include <loco/IR/Node.h>

namespace locoex
{

/// @brief enumeration of mixin class
enum class CircleNodeTrait
{
  FusedActFunc,
};

template <CircleNodeTrait T> class CircleNodeMixin;

template <> class CircleNodeMixin<CircleNodeTrait::FusedActFunc>
{
public:
  CircleNodeMixin() = default;

public:
  FusedActFunc fusedActivationFunction() const { return _fused_act_fun; }
  void fusedActivationFunction(FusedActFunc fused_act_fun) { _fused_act_fun = fused_act_fun; }

private:
  FusedActFunc _fused_act_fun = FusedActFunc::UNDEFINED;
};

/**
 * @brief INSTANCE_NORM in circle
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

} // namespace locoex

#endif // __LOCOEX_IR_CIRCLENODES_H__
