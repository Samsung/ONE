/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "StringUtils.h"

#include <luci/IR/CircleNodeDecl.h>

#include <cassert>

namespace dalgona
{

const std::string toString(luci::CircleOpcode opcode)
{
  static const char *names[] = {
#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) #CIRCLE_CLASS,
#define CIRCLE_VNODE(OPCODE, CIRCLE_CLASS) #CIRCLE_CLASS,
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE
#undef CIRCLE_VNODE
  };

  auto const node_name = names[static_cast<int>(opcode)];

  assert(std::string(node_name).substr(0, 6) == "Circle"); // FIX_ME_UNLESS

  // Return substring of class name ("Circle" is sliced out)
  // Ex: Return "Conv2D" for "CircleConv2D" node
  return std::string(node_name).substr(6);
}

const std::string toString(luci::FusedActFunc fused_act)
{
  switch (fused_act)
  {
    case (luci::FusedActFunc::UNDEFINED):
      return std::string("undefined");
    case (luci::FusedActFunc::NONE):
      return std::string("none");
    case (luci::FusedActFunc::RELU):
      return std::string("relu");
    case (luci::FusedActFunc::RELU_N1_TO_1):
      return std::string("relu_n1_to_1");
    case (luci::FusedActFunc::RELU6):
      return std::string("relu6");
    case (luci::FusedActFunc::TANH):
      return std::string("tanh");
    case (luci::FusedActFunc::SIGN_BIT):
      return std::string("sign_bit");
    default:
      throw std::runtime_error("Unsupported activation function");
  }
}

} // namespace dalgona
