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

#ifndef __LOCO_IR_NODE_POOL_H__
#define __LOCO_IR_NODE_POOL_H__

#include "loco/IR/Node.h"
#include "loco/IR/Graph.forward.h"

#include "loco/ADT/ObjectPool.h"

namespace loco
{

class NodePool final : public ObjectPool<Node>
{
public:
  friend class Graph;

public:
  ~NodePool();

public:
  template <typename Derived, typename... Args> Derived *create(Args &&...args)
  {
    std::unique_ptr<Derived> ptr{new Derived(std::forward<Args>(args)...)};
    ptr->graph(_graph);
    return ObjectPool<Node>::take<Derived>(std::move(ptr));
  }

  void destroy(Node *node)
  {
    if (!ObjectPool<Node>::erase(node))
    {
      throw std::invalid_argument{"node"};
    }
  }

private:
  /// Only "Graph" is permitted to invoke this private method.
  void graph(Graph *g) { _graph = g; }

private:
  Graph *_graph = nullptr;
};

} // namespace loco

#endif // __LOCO_IR_NODE_POOL_H__
