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

#include "luci/Pass/QuantizePreCheckerPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

#include <luci/Log.h>

namespace luci
{

namespace
{

void check_const_opcode(luci::CircleNode *node)
{
  if (node == nullptr)
    return;

  if (node->opcode() != luci::CircleOpcode::CIRCLECONST and
      node->opcode() != luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE)
  {
    throw std::runtime_error("Unsupported non const input " + node->name());
  }
}

struct ConstInputChecker final : public luci::CircleNodeMutableVisitor<void>
{
// INPUT_NAME is name for input const for current NODE
#define CHECK_NODE_WITH_ONE_INPUT_CONST(NODE, INPUT_NAME)                    \
  void visit(NODE *node)                                                     \
  {                                                                          \
    const auto input = dynamic_cast<luci::CircleNode *>(node->INPUT_NAME()); \
    check_const_opcode(input);                                               \
  }

// INPUT_NAME_1 and INPUT_NAME_2 are names for input const for current NODE
#define CHECK_NODE_WITH_TWO_INPUT_CONST(NODE, INPUT_NAME_1, INPUT_NAME_2)        \
  void visit(NODE *node)                                                         \
  {                                                                              \
    const auto input_1 = dynamic_cast<luci::CircleNode *>(node->INPUT_NAME_1()); \
    const auto input_2 = dynamic_cast<luci::CircleNode *>(node->INPUT_NAME_2()); \
                                                                                 \
    check_const_opcode(input_1);                                                 \
    check_const_opcode(input_2);                                                 \
  }

// INPUT_NAME_1, INPUT_NAME_2 and INPUT_NAME_3 are names for input const for current NODE
#define CHECK_NODE_WITH_THREE_INPUT_CONST(NODE, INPUT_NAME_1, INPUT_NAME_2, INPUT_NAME_3) \
  void visit(NODE *node)                                                                  \
  {                                                                                       \
    const auto input_1 = dynamic_cast<luci::CircleNode *>(node->INPUT_NAME_1());          \
    const auto input_2 = dynamic_cast<luci::CircleNode *>(node->INPUT_NAME_2());          \
    const auto input_3 = dynamic_cast<luci::CircleNode *>(node->INPUT_NAME_3());          \
                                                                                          \
    check_const_opcode(input_1);                                                          \
    check_const_opcode(input_2);                                                          \
    check_const_opcode(input_3);                                                          \
  }

  // Skip other circle node
  void visit(luci::CircleNode *) {}

  // Ops that receive one const nodes as inputs
  CHECK_NODE_WITH_ONE_INPUT_CONST(luci::CirclePRelu, alpha)
  CHECK_NODE_WITH_ONE_INPUT_CONST(luci::CircleRmsNorm, gamma)

  // Ops that receive two const node as an inputs
  CHECK_NODE_WITH_TWO_INPUT_CONST(luci::CircleConv2D, filter, bias)
  CHECK_NODE_WITH_TWO_INPUT_CONST(luci::CircleDepthwiseConv2D, filter, bias)
  CHECK_NODE_WITH_TWO_INPUT_CONST(luci::CircleFullyConnected, weights, bias)
  CHECK_NODE_WITH_TWO_INPUT_CONST(luci::CircleInstanceNorm, gamma, beta)

  // Ops that receive three const nodes as an inputs
  CHECK_NODE_WITH_THREE_INPUT_CONST(luci::CircleTransposeConv, inputSizes, filter, bias)

#undef CHECK_NODE_WITH_ONE_INPUT_CONST
#undef CHECK_NODE_WITH_TWO_INPUT_CONST
#undef CHECK_NODE_WITH_THREE_INPUT_CONST
};

} // namespace

/**
 * Verify the input model has the form acceptable by quantizer
 */
bool QuantizePreCheckerPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizePreCheckerPass Start" << std::endl;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    // Check const inputs
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    ConstInputChecker checker{};
    circle_node->accept(&checker);
  }

  INFO(l) << "QuantizePreCheckerPass End" << std::endl;

  return false; // one time run
}

} // namespace luci
