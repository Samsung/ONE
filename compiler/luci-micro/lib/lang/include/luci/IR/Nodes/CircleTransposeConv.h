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

#ifndef __LUCI_IR_CIRCLETRANSPOSECONV_H__
#define __LUCI_IR_CIRCLETRANSPOSECONV_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrPadding.h"
#include "luci/IR/AttrStride.h"
#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief TRANSPOSE_CONV in Circle
 *
 * @note  Argument node function names are from TensorFlow. So referring 'in' and
 *        'out' acutally means 'out' and 'in' of the this node.
 */
class CircleTransposeConv final
    : public FixedArityNode<3, CircleNodeImpl<CircleOpcode::TRANSPOSE_CONV>>
{
public:
  loco::Node *inputSizes(void) const { return at(0)->node(); }
  void inputSizes(Node *node) { at(0)->node(node); }

  loco::Node *filter(void) const { return at(1)->node(); }
  void filter(Node *node) { at(1)->node(node); }

  loco::Node *outBackprop(void) const { return at(2)->node(); }
  void outBackprop(Node *node) { at(2)->node(node); }

public:
  const Padding &padding(void) const { return _padding; }
  void padding(const Padding &padding) { _padding = padding; }

  const Stride *stride(void) const { return &_stride; }
  Stride *stride(void) { return &_stride; }

private:
  Padding _padding{Padding::UNDEFINED};
  Stride _stride;
};

} // namespace luci

#endif // __LUCI_IR_CIRCLETRANSPOSECONV_H__
