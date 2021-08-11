/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_CIRCLE_NODE_MIXINS_H__
#define __LUCI_IR_CIRCLE_NODE_MIXINS_H__

#include "luci/IR/AttrFusedActFunc.h"

#include <loco/IR/Node.h>
#include <loco/IR/NodeMixins.h>

#include <vector>

namespace luci
{

/// @brief enumeration of mixin class
enum class CircleNodeTrait
{
  FusedActFunc,
  Bias
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
 * @brief Mixin class for nodes that has a bias input
 */
template <> class CircleNodeMixin<CircleNodeTrait::Bias>
{
public:
  CircleNodeMixin() = default;

public:
  virtual loco::Node *bias(void) const = 0; /// @brief get the input for bias.
  virtual void bias(loco::Node *node) = 0;  /// @brief set the input for bias.
};

/**
 * @brief Nodes with the fixed number of inputs
 *
 * TODO Deprecated this class, and use loco::FixedArity instead
 */
template <uint32_t N, typename Base> class FixedArityNode : public Base
{
public:
  FixedArityNode()
  {
    _args.resize(N);
    for (uint32_t n = 0; n < N; ++n)
    {
      _args[n] = std::make_unique<loco::Use>(this);
    }
  }

  virtual ~FixedArityNode() = default;

public:
  uint32_t arity(void) const final { return N; }

  loco::Node *arg(uint32_t n) const final { return _args.at(n)->node(); }

  void drop(void) final
  {
    for (uint32_t n = 0; n < N; ++n)
    {
      _args.at(n)->node(nullptr);
    }
  }

protected:
  // This API allows inherited classes to access "_args" field.
  loco::Use *at(uint32_t n) const { return _args.at(n).get(); }

private:
  std::vector<std::unique_ptr<loco::Use>> _args{};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_NODE_MIXINS_H__
