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

#ifndef __LUCI_PARTITION_CONNECT_NODE_H__
#define __LUCI_PARTITION_CONNECT_NODE_H__

#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

/**
 * @note MapNode2Clone is used as a map from original node to cloned node
 *       to find input of a cloned node
 *
 *   (Original)              (Clone)
 *
 *     [A]                  [A']
 *      |   [B]              |   [B']
 *      |    |               |    |
 *       \  /                 \  /
 *        [C]                 [C']
 *
 *  From view of [C'] we need to find [A'] and [B']. We know [C] from [C'],
 *  then we can get from input of [C] as [A], [B] then [A]->[A'] and [B]->[B']
 *  from the map.
 */
using MapNode2Clone = std::map<const CircleNode * /* ORG */, CircleNode * /* CLONE */>;

struct CloneContext
{
  std::pair<MapNode2Clone::iterator, bool> emplace(const CircleNode *org, CircleNode *clone)
  {
    return node2clone.emplace(org, clone);
  }
  MapNode2Clone::iterator find(const CircleNode *org) { return node2clone.find(org); }
  MapNode2Clone::iterator end(void) { return node2clone.end(); }

  MapNode2Clone node2clone;
};

class ConnectNode final : public luci::CircleNodeVisitor<void>
{
public:
  ConnectNode(luci::CloneContext &clonecontext) : _clonecontext(clonecontext){};

public:
  void visit(const luci::CircleAdd *) final;
  void visit(const luci::CircleConst *) final;
  void visit(const luci::CircleDiv *) final;
  void visit(const luci::CircleMean *) final;
  void visit(const luci::CircleMul *) final;
  void visit(const luci::CirclePow *) final;
  void visit(const luci::CircleRsqrt *) final;
  void visit(const luci::CircleSqrt *) final;
  void visit(const luci::CircleSquaredDifference *) final;
  void visit(const luci::CircleSub *) final;

protected:
  luci::CircleNode *find_clone(const luci::CircleNode *node);

protected:
  luci::CloneContext &_clonecontext;
};

} // namespace luci

#endif // __LUCI_PARTITION_CONNECT_NODE_H__
