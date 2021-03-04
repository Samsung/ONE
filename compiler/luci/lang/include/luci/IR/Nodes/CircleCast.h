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

#ifndef __LUCI_IR_CIRCELECAST_H__
#define __LUCI_IR_CIRCELECAST_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief CAST in Circle
 */
class CircleCast final : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::CAST>>
{
public:
  loco::Node *x(void) const { return at(0)->node(); }
  void x(loco::Node *node) { at(0)->node(node); }

public:
  loco::DataType in_data_type(void) const { return _in_data_type; }
  void in_data_type(loco::DataType it) { _in_data_type = it; }

  loco::DataType out_data_type(void) const { return _out_data_type; }
  void out_data_type(loco::DataType ot) { _out_data_type = ot; }

private:
  loco::DataType _in_data_type{loco::DataType::FLOAT32};
  loco::DataType _out_data_type{loco::DataType::FLOAT32};
};

} // namespace luci

#endif // __LUCI_IR_CIRCELECAST_H__
