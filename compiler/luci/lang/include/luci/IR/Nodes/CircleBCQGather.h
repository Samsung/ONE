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

#ifndef __LUCI_IR_CIRCLEBCQGATHER_H__
#define __LUCI_IR_CIRCLEBCQGATHER_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief BCQ_GATHER in Circle
 */
class CircleBCQGather final : public FixedArityNode<4, CircleNodeImpl<CircleOpcode::BCQ_GATHER>>
{
public:
  loco::Node *input_scales(void) const { return at(0)->node(); }
  void input_scales(loco::Node *node) { at(0)->node(node); }

  loco::Node *input_binary(void) const { return at(1)->node(); }
  void input_binary(loco::Node *node) { at(1)->node(node); }

  loco::Node *indices(void) const { return at(2)->node(); }
  void indices(loco::Node *node) { at(2)->node(node); }

  loco::Node *input_clusters(void) const { return at(3)->node(); }
  void input_clusters(loco::Node *node) { at(3)->node(node); }

public:
  int32_t axis(void) const { return _axis; }
  void axis(int32_t axis) { _axis = axis; }

  int32_t input_hidden_size(void) const { return _input_hidden_size; }
  void input_hidden_size(int32_t input_hidden_size) { _input_hidden_size = input_hidden_size; }

private:
  int32_t _axis{0};
  int32_t _input_hidden_size{0};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEBCQGATHER_H__
