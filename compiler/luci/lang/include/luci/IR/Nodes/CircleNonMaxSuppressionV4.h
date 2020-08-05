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

#ifndef __LUCI_IR_CIRCLE_NON_MAX_SUPPRESSION_V4_H__
#define __LUCI_IR_CIRCLE_NON_MAX_SUPPRESSION_V4_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/LuciNodeMixins.h"

namespace luci
{

/**
 * @brief NON_MAX_SUPPRESSION_V4 in Circle
 */
class CircleNonMaxSuppressionV4 final
    : public FixedArityNode<5, CircleNodeImpl<CircleOpcode::NON_MAX_SUPPRESSION_V4>>
{
public:
  loco::Node *boxes(void) const { return at(0)->node(); }
  void boxes(loco::Node *node) { at(0)->node(node); }

  loco::Node *scores(void) const { return at(1)->node(); }
  void scores(loco::Node *node) { at(1)->node(node); }

  loco::Node *max_output_size(void) const { return at(2)->node(); }
  void max_output_size(loco::Node *node) { at(2)->node(node); }

  loco::Node *iou_threshold(void) const { return at(3)->node(); }
  void iou_threshold(loco::Node *node) { at(3)->node(node); }

  loco::Node *score_threshold(void) const { return at(4)->node(); }
  void score_threshold(loco::Node *node) { at(4)->node(node); }

public:
  bool pad_to_max_output_size(void) const { return _pad_to_max_output_size; }
  void pad_to_max_output_size(bool pad_to_max_output_size)
  {
    _pad_to_max_output_size = pad_to_max_output_size;
  }

private:
  bool _pad_to_max_output_size{false};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_NON_MAX_SUPPRESSION_V4_H__
