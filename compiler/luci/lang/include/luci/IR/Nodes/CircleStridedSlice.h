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

#ifndef __LUCI_IR_STRIDEDSLICE_H__
#define __LUCI_IR_STRIDEDSLICE_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief STRIDED_SLICE in Circle
 */
class CircleStridedSlice final
    : public FixedArityNode<4, CircleNodeImpl<CircleOpcode::STRIDED_SLICE>>
{
public:
  loco::Node *input(void) const { return at(0)->node(); }
  void input(loco::Node *node) { at(0)->node(node); }

  loco::Node *begin(void) const { return at(1)->node(); }
  void begin(loco::Node *node) { at(1)->node(node); }

  loco::Node *end(void) const { return at(2)->node(); }
  void end(loco::Node *node) { at(2)->node(node); }

  loco::Node *strides(void) const { return at(3)->node(); }
  void strides(loco::Node *node) { at(3)->node(node); }

public:
  int32_t begin_mask() const { return _begin_mask; }
  void begin_mask(int32_t mask) { _begin_mask = mask; }

  int32_t end_mask() const { return _end_mask; }
  void end_mask(int32_t mask) { _end_mask = mask; }

  int32_t ellipsis_mask() const { return _ellipsis_mask; }
  void ellipsis_mask(int32_t mask) { _ellipsis_mask = mask; }

  int32_t new_axis_mask() const { return _new_axis_mask; }
  void new_axis_mask(int32_t mask) { _new_axis_mask = mask; }

  int32_t shrink_axis_mask() const { return _shrink_axis_mask; }
  void shrink_axis_mask(int32_t mask) { _shrink_axis_mask = mask; }

private:
  int32_t _begin_mask{0};
  int32_t _end_mask{0};
  int32_t _ellipsis_mask{0};
  int32_t _new_axis_mask{0};
  int32_t _shrink_axis_mask{0};
};

} // namespace luci

#endif // __LUCI_IR_STRIDEDSLICE_H__
