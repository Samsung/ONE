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

#ifndef __LUCI_IR_CIRCELSPARSETODENSE_H__
#define __LUCI_IR_CIRCELSPARSETODENSE_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief SPARSE_TO_DENSE in Circle
 */
class CircleSparseToDense final
    : public FixedArityNode<4, CircleNodeImpl<CircleOpcode::SPARSE_TO_DENSE>>
{
public:
  loco::Node *indices(void) const { return at(0)->node(); }
  void indices(loco::Node *node) { at(0)->node(node); }

  loco::Node *output_shape(void) const { return at(1)->node(); }
  void output_shape(loco::Node *node) { at(1)->node(node); }

  loco::Node *values(void) const { return at(2)->node(); }
  void values(loco::Node *node) { at(2)->node(node); }

  loco::Node *default_value(void) const { return at(3)->node(); }
  void default_value(loco::Node *node) { at(3)->node(node); }

public:
  bool validate_indices(void) const { return _validate_indices; }
  void validate_indices(bool validate_indices) { _validate_indices = validate_indices; }

private:
  bool _validate_indices{true};
};

} // namespace luci

#endif // __LUCI_IR_CIRCELSPARSETODENSE_H__
