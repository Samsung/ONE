/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loco/IR/Algorithm.h"

#include <cassert>
#include <set>
#include <stack>

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

  loco::Node &node(void) const { return *_ptr; }

  void advance(void) { _pos += 1; }

private:
  loco::Node *_ptr = nullptr;
  int64_t _pos = -1;
};

} // namespace

namespace loco
{

// TODO Support cyclic graphs
std::vector<loco::Node *> postorder_traversal(const std::vector<loco::Node *> &roots)
{
  std::vector<loco::Node *> res;

  std::set<loco::Node *> visited_nodes;
  std::stack<Frame> frames;

  auto visited = [&visited_nodes](loco::Node *node) {
    return visited_nodes.find(node) != visited_nodes.end();
  };

  // NOTE There is not much difference between "auto" and "auto &" as node is of "loco::Node *"
  // type.
  for (auto node : roots)
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

    if (top_frame.pos() < static_cast<int64_t>(top_frame.node().arity()))
    {
      // Let's visit the next argument
      //
      // NOTE "next" may be nullptr if a graph is under construction.
      if (auto next = top_frame.node().arg(top_frame.pos()))
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

std::set<loco::Node *> active_nodes(const std::vector<loco::Node *> &roots)
{
  // This implementation works but may be inefficient
  //
  // TODO Use efficient implementation if necessary
  auto nodes = postorder_traversal(roots);
  return std::set<loco::Node *>{nodes.begin(), nodes.end()};
}

} // namespace loco
