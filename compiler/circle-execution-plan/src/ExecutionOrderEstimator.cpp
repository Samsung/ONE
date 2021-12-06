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

#include "ExecutionOrderEstimator.h"

namespace circle_planner
{
namespace
{

class Frame final
{
public:
  Frame(loco::Node *ptr) : _ptr{ptr}, _pos{-1}
  {
    // DO NOTHING
  }

public:
  loco::Node *ptr(void) const { return _ptr; }
  int64_t pos(void) const { return _pos; }

  void advance(void) { _pos += 1; }

private:
  loco::Node *_ptr = nullptr;
  int64_t _pos = -1;
};

bool isExecutableNode(const luci::CircleNode *node)
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
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    case luci::CircleOpcode::CIRCLEWHILEOUT:
      return false;
    default:
      return true;
  }
}

bool NextSet(std::vector<uint32_t> &vector)
{
  int n = vector.size();
  int j;
  do
  {
    j = n - 2;
    while (j != -1 && vector[j] >= vector[j + 1])
      j--;

    if (j == -1)
      return false;

    int k = n - 1;
    while (vector[j] >= vector[k])
      k--;

    std::iter_swap(vector.begin() + j, vector.begin() + k);
    int l = j + 1, r = n - 1;

    while (l < r)
      std::iter_swap(vector.begin() + (l++), vector.begin() + (r--));
  } while (j > n - 1);
  return true;
}

} // namespace

void ExecutionOrderEstimator::get_node_order_inform_vector(
  std::vector<NodeOrderInform> &node_order_inform_vector)
{
  std::set<loco::Node *> visited_nodes;
  std::stack<Frame> frames;

  auto visited = [&visited_nodes](loco::Node *node) {
    return visited_nodes.find(node) != visited_nodes.end();
  };

  for (auto node : _root_nodes)
  {
    assert((node != nullptr) && "root is invalid");
    frames.push(Frame{node});
  }

  while (!frames.empty())
  {
    auto &top_frame = frames.top();

    if (top_frame.pos() == -1)
    {
      if (visited(top_frame.ptr()))
      {
        frames.pop();
        continue;
      }
      visited_nodes.insert(top_frame.ptr());

      auto exec_nodes_count = 0;
      for (int i = 0; i < top_frame.ptr()->arity(); ++i)
      {
        if (auto next = top_frame.ptr()->arg(i))
          if (isExecutableNode(loco::must_cast<luci::CircleNode *>(next)))
            exec_nodes_count++;
      }
      if (exec_nodes_count > 1)
      {
        NodeOrderInform node_order_inform;
        node_order_inform._node = top_frame.ptr();
        node_order_inform_vector.emplace_back(node_order_inform);
      }
    }

    top_frame.advance();

    assert(top_frame.pos() >= 0);

    auto it = std::find_if(node_order_inform_vector.begin(), node_order_inform_vector.end(),
                           [top_frame](const auto &node_order_inform) {
                             return node_order_inform._node == top_frame.ptr();
                           });

    if (top_frame.pos() < static_cast<int64_t>(top_frame.ptr()->arity()))
    {
      if (auto next = top_frame.ptr()->arg(top_frame.pos()))
      {
        if (it != node_order_inform_vector.end())
        {
          if (isExecutableNode(loco::must_cast<luci::CircleNode *>(next)))
          {
            it->executable_arg_idx.emplace_back(top_frame.pos());
          }
          else
          {
            it->non_executable_arg_idx.emplace_back(top_frame.pos());
          }
        }
        frames.push(Frame{next});
      }
    }
    else
    {
      // Let's visit the current argument (all the arguments are already visited)
      if (it != node_order_inform_vector.end())
      {
        do
        {
          auto combination = it->executable_arg_idx;
          for (unsigned int &s : it->non_executable_arg_idx)
            combination.emplace_back(s);
          it->all_combinations.emplace_back(combination);
        } while (NextSet(it->executable_arg_idx));
      }
      frames.pop();
    }
  }
}

std::vector<loco::Node *> ExecutionOrderEstimator::get_current_order(
  const std::vector<NodeOrderInform> &node_order_inform_vector)
{
  std::vector<loco::Node *> res;

  std::set<loco::Node *> visited_nodes;
  std::stack<Frame> frames;

  auto visited = [&visited_nodes](loco::Node *node) {
    return visited_nodes.find(node) != visited_nodes.end();
  };

  // NOTE There is not much difference between "auto" and "auto &" as node is of "loco::Node *"
  // type.
  for (auto node : _root_nodes)
  {
    assert((node != nullptr) && "root is invalid");
    frames.push(Frame{node});
  }

  while (!frames.empty())
  {
    auto &top_frame = frames.top();

    if (top_frame.pos() == -1)
    {
      if (visited(top_frame.ptr()))
      {
        frames.pop();
        continue;
      }
      visited_nodes.insert(top_frame.ptr());
    }

    top_frame.advance();

    assert(top_frame.pos() >= 0);

    if (top_frame.pos() < static_cast<int64_t>(top_frame.ptr()->arity()))
    {
      // Let's visit the next argument
      //
      // NOTE "next" may be nullptr if a graph is under construction.
      auto idx = top_frame.pos();

      auto it = std::find_if(
        node_order_inform_vector.begin(), node_order_inform_vector.end(),
        [top_frame](auto &node_inform) { return node_inform._node == top_frame.ptr(); });
      if (it != node_order_inform_vector.end())
      {
        idx = it->all_combinations.at(it->current_idx).at(idx);
      }

      if (auto next = top_frame.ptr()->arg(idx))
      {
        frames.push(Frame{next});
      }
    }
    else
    {
      // Let's visit the current argument (all the arguments are already visited)
      auto curr = top_frame.ptr();
      res.emplace_back(curr);
      frames.pop();
    }
  }
  return res;
}

} // namespace circle_planner
