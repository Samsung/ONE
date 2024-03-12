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

#ifndef __LUCI_IR_CIRCLE_CIR_GRU_H__
#define __LUCI_IR_CIRCLE_CIR_GRU_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief GRU in Circle
 */
class CircleCirGru final : public FixedArityNode<6, CircleNodeImpl<CircleOpcode::CIR_GRU>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *hidden_hidden(void) const { return at(1)->node(); }
  void hidden_hidden(loco::Node *node) { at(1)->node(node); }

  loco::Node *hidden_hidden_bias(void) const { return at(2)->node(); }
  void hidden_hidden_bias(loco::Node *node) { at(2)->node(node); }

  loco::Node *hidden_input(void) const { return at(3)->node(); }
  void hidden_input(loco::Node *node) { at(3)->node(node); }

  loco::Node *hidden_input_bias(void) const { return at(4)->node(); }
  void hidden_input_bias(loco::Node *node) { at(4)->node(node); }

  loco::Node *state(void) const { return at(5)->node(); }
  void state(loco::Node *node) { at(5)->node(node); }

public:
  FusedActFunc fusedActivationFunction() const { return _fused_act_fun; }
  void fusedActivationFunction(FusedActFunc fused_act_fun) { _fused_act_fun = fused_act_fun; }

  bool returnSequences() const { return _return_sequences; }
  void returnSequences(bool return_sequences) { _return_sequences = return_sequences; }

  bool timeMajor() const { return _time_major; }
  void timeMajor(bool time_major) { _time_major = time_major; }

private:
  FusedActFunc _fused_act_fun = FusedActFunc::NONE;
  bool _return_sequences = false;
  bool _time_major = false;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_CIR_GRU_H__
