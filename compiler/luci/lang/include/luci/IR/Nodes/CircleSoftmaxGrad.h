/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_CIRCLESOFTMAX_GRAD_H__
#define __LUCI_IR_CIRCLESOFTMAX_GRAD_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief SOFTMAX GRAD in Circle
 */
class CircleSoftmaxGrad final : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::SOFTMAX_GRAD>>
{
public:
  loco::Node *softmax_values(void) const { return at(0)->node(); }
  void softmax_values(loco::Node *node) { at(0)->node(node); }
};

} // namespace luci

#endif // __LUCI_IR_CIRCLESOFTMAX_GRAD_H__
