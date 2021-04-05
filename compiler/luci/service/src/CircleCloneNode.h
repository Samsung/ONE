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

#ifndef __CIRCLE_CLONE_NODE_H__
#define __CIRCLE_CLONE_NODE_H__

#include <luci/IR/CircleNodes.h>

#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

class CloneNode final : public luci::CircleNodeVisitor<luci::CircleNode *>
{
public:
  CloneNode(loco::Graph *graph) : _graph(graph){};

public:
  luci::CircleNode *visit(const luci::CircleAdd *) final;
  luci::CircleNode *visit(const luci::CircleConst *) final;
  luci::CircleNode *visit(const luci::CircleDiv *) final;
  luci::CircleNode *visit(const luci::CircleMean *) final;
  luci::CircleNode *visit(const luci::CircleMul *) final;
  luci::CircleNode *visit(const luci::CirclePow *) final;
  luci::CircleNode *visit(const luci::CircleRsqrt *) final;
  luci::CircleNode *visit(const luci::CircleSqrt *) final;
  luci::CircleNode *visit(const luci::CircleSquaredDifference *) final;
  luci::CircleNode *visit(const luci::CircleSub *) final;
  // TODO add all nodes

  // NOTE CircleNodeVisitor will throw if not supported here

protected:
  loco::Graph *_graph = nullptr;
};

} // namespace luci

#endif // __CIRCLE_CLONE_NODE_H__
