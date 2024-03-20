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

#ifndef __LUCI_IR_CIRCLEMAXPOOL2D_GRAD_H__
#define __LUCI_IR_CIRCLEMAXPOOL2D__GRAD_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrFilter.h"
#include "luci/IR/AttrPadding.h"
#include "luci/IR/AttrStride.h"
#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief MAX_POOL_2D in Circle
 */
class CircleMaxPool2DGrad final : public FixedArityNode<2, CircleNodeImpl<CircleOpcode::MAX_POOL_2D_GRAD>>,
                                  public CircleNodeMixin<CircleNodeTrait::FusedActFunc>
{
public:
  loco::Node *input_grad(void) const { return at(0)->node(); }
  void input_grad(loco::Node *node) { at(0)->node(node); }

  loco::Node *input_activations(void) const { return at(1)->node(); }
  void input_activations(loco::Node *node) { at(1)->node(node); }

public:
  Padding padding() const { return _padding; }
  void padding(Padding padding) { _padding = padding; }

  const Filter *filter(void) const { return &_filter; }
  Filter *filter(void) { return &_filter; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

private:
  Padding _padding{Padding::UNDEFINED};
  Stride _stride;
  Filter _filter;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEMAXPOOL2D__GRAD_H__
