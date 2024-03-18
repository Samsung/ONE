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

#ifndef __LUCI_IR_CIRCLE_REDUCE_SUM_H__
#define __LUCI_IR_CIRCLE_REDUCE_SUM_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/CircleNodeMixins.h"

namespace luci
{

/**
 * @brief REDUCE_SUM in Circle
 */
class CircleReduceSum final : public FixedArityNode<2, CircleNodeImpl<CircleOpcode::REDUCE_SUM>>
  {
  public:
    loco::Node *input(void) const { return at(0)->node(); }
    void input(loco::Node *node) { at(0)->node(node); }

    loco::Node *reduction_indices(void) const { return at(1)->node(); }
    void reduction_indices(loco::Node *node) { at(1)->node(node); }

  public:
    bool keep_dims(void) const { return _keep_dims; }
    void keep_dims(bool keep_dims) { _keep_dims = keep_dims; }

  private:
    bool _keep_dims{false};
  };

} // namespace luci

#endif // __LUCI_IR_CIRCLE_REDUCE_SUM_H__
