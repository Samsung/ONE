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

#ifndef __LUCI_IR_CIRCLEMAXPOOL2D_H__
#define __LUCI_IR_CIRCLEMAXPOOL2D_H__

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
class CircleMaxPool2D final : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::MAX_POOL_2D>>,
                              public CircleNodeMixin<CircleNodeTrait::FusedActFunc>
{
public:
  CircleMaxPool2D() : _padding(Padding::UNDEFINED)
  { /* empty */
  }

public:
  loco::Node *value(void) const { return at(0)->node(); }
  void value(loco::Node *node) { at(0)->node(node); }

  Padding padding() const { return _padding; }
  void padding(Padding padding) { _padding = padding; }

  const Filter *filter(void) const { return &_filter; }
  Filter *filter(void) { return &_filter; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

private:
  Padding _padding;
  Stride _stride;
  Filter _filter;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEMAXPOOL2D_H__
