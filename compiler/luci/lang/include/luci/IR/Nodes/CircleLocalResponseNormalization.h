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

#ifndef __LUCI_IR_CIRCLELOCAL_RESPONSE_NORMALIZATION_H__
#define __LUCI_IR_CIRCLELOCAL_RESPONSE_NORMALIZATION_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief LOCAL_RESPONSE_NORMALIZATION in Circle
 */
class CircleLocalResponseNormalization final
  : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::LOCAL_RESPONSE_NORMALIZATION>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

public:
  int32_t radius(void) const { return _radius; }
  void radius(int32_t radius) { _radius = radius; }

  float bias(void) const { return _bias; }
  void bias(float bias) { _bias = bias; }

  float alpha(void) const { return _alpha; }
  void alpha(float alpha) { _alpha = alpha; }

  float beta(void) const { return _beta; }
  void beta(float beta) { _beta = beta; }

private:
  int32_t _radius{5};
  float _bias{1.0f};
  float _alpha{1.0f};
  float _beta{0.5f};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLELOCAL_RESPONSE_NORMALIZATION_H__
