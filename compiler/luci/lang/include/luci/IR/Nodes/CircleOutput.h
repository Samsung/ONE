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

#ifndef __LUCI_IR_CIRCLEOUTPUT_H__
#define __LUCI_IR_CIRCLEOUTPUT_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/AttrFusedActFunc.h"
#include "luci/IR/LuciNodeMixins.h"

#include <loco/IR/GraphOutputIndex.h>

namespace luci
{

/**
 * @brief CircleNode for Output of the Graph
 * @note  This will not be exported as a specific op
 */
class CircleOutput final : public FixedArityNode<1, CircleNodeImpl<CircleOpcode::CIRCLEOUTPUT>>
{
public:
  CircleOutput() = default;

  void index(const loco::GraphOutputIndex &index);
  loco::GraphOutputIndex index(void) const;

  bool indexed(void) const { return _index != -1; }

public:
  loco::Node *from(void) const { return at(0)->node(); }
  void from(loco::Node *node) { at(0)->node(node); }

private:
  int64_t _index = -1; // Uninitialized
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEOUTPUT_H__
