/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef CIRCLE_EXECUTION_PLAN_HELPER_H
#define CIRCLE_EXECUTION_PLAN_HELPER_H

#include <luci/IR/CircleNode.h>

namespace circle_planner
{
// struct for execution order estimation
struct NodeOrderInform
{
  // current node
  loco::Node *_node = nullptr;
  // indexes of non-executable arguments (non-executable nodes parents in graph) of current node
  std::vector<uint32_t> non_executable_arg_idx;
  // indexes of executable arguments (executable nodes parents in graph) of current node
  std::vector<uint32_t> executable_arg_idx;
  // vector of combinations of arguments of current node: different variants of the order
  // in which we will visit the arguments.
  // note: combinations are considered only for executable nodes
  std::vector<std::vector<uint32_t>> all_combinations;

  // the index that stores the number of the current combination in all_combinations vector
  uint32_t current_idx = 0;
};

inline bool isMultiOutputNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // The following nodes denote outputs of multiple-output nodes.
    // The list is synchronized with the same list from luci-interpreter/src/loader/GraphLoader.cpp
    case luci::CircleOpcode::CIRCLEIFOUT:
    case luci::CircleOpcode::CIRCLESPLITOUT:
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    case luci::CircleOpcode::CIRCLEWHILEOUT:
      return false;
    default:
      return true;
  }
}

inline bool isExecutableNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // These nodes denote inputs / outputs of a graph.
    case luci::CircleOpcode::CIRCLECONST:
    case luci::CircleOpcode::CIRCLEINPUT:
    case luci::CircleOpcode::CIRCLEOUTPUT:
    case luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE:
      // The following nodes denote outputs of multiple-output nodes.
    case luci::CircleOpcode::CIRCLEIFOUT:
    case luci::CircleOpcode::CIRCLESPLITOUT:
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    case luci::CircleOpcode::CIRCLEWHILEOUT:
      return false;
    default:
      return true;
  }
}

inline bool isTensorProducingNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // The following nodes are multiple-output nodes. They do not produce tensors, the tensors
    // are produced by the corresponding *Out nodes instead.
    // The list is synchronized with the same list from luci-interpreter/src/loader/GraphLoader.cpp
    case luci::CircleOpcode::IF:
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::UNPACK:
      return false;
    default:
      return true;
  }
}

} // namespace circle_planner

#endif // CIRCLE_EXECUTION_PLAN_HELPER_H
