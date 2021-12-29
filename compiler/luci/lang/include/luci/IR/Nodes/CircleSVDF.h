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

#ifndef __LUCI_IR_CIRCLE_SVDF_H__
#define __LUCI_IR_CIRCLE_SVDF_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief SVDF in Circle
 */
class CircleSVDF final : public FixedArityNode<5, CircleNodeImpl<CircleOpcode::SVDF>>,
                         public CircleNodeMixin<CircleNodeTrait::FusedActFunc>
{
public:
  CircleSVDF() = default;

public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *weight_feature(void) const { return at(1)->node(); }
  void weight_feature(loco::Node *node) { at(1)->node(node); }

  loco::Node *weight_time(void) const { return at(2)->node(); }
  void weight_time(loco::Node *node) { at(2)->node(node); }

  loco::Node *bias(void) const { return at(3)->node(); }
  void bias(loco::Node *node) { at(3)->node(node); }

  loco::Node *input_activation_state(void) const { return at(4)->node(); }
  void input_activation_state(loco::Node *node) { at(4)->node(node); }

public:
  bool asymmetric_quantize_inputs() const { return _asymmetric_quantize_inputs; }
  void asymmetric_quantize_inputs(bool asymmetric_quantize_inputs)
  {
    _asymmetric_quantize_inputs = asymmetric_quantize_inputs;
  }

  int32_t svdf_rank() const { return _rank; }
  void svdf_rank(int32_t svdf_rank) { _rank = svdf_rank; }

private:
  bool _asymmetric_quantize_inputs = false;
  int32_t _rank = 0;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_SVDF_H__
