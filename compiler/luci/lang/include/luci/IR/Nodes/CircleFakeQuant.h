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

#ifndef __LUCI_IR_CIRCLE_FAKE_QUANT_H__
#define __LUCI_IR_CIRCLE_FAKE_QUANT_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief FAKE_QUANT in Circle
 * @note  'inputs' came from TF.quantize.fake_quant_from_min_max_vars
 */
class CircleFakeQuant final : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::FAKE_QUANT>>
{
public:
  loco::Node *inputs(void) const { return at(0)->node(); }
  void inputs(loco::Node *node) { at(0)->node(node); }

public:
  float min(void) const { return _min; }
  void min(float min) { _min = min; }

  float max(void) const { return _max; }
  void max(float max) { _max = max; }

  int32_t num_bits(void) const { return _num_bits; }
  void num_bits(int32_t num_bits) { _num_bits = num_bits; }

  bool narrow_range(void) const { return _narrow_range; }
  void narrow_range(bool narrow_range) { _narrow_range = narrow_range; }

private:
  float _min{0.0f};
  float _max{0.0f};
  int32_t _num_bits{0};
  bool _narrow_range{false};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEGATHER_H__
