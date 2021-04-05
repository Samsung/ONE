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

#include "CircleNodeConnect.h"

#include <oops/UserExn.h>

namespace
{

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
  // TODO add all nodes

protected:
  luci::CircleNode *find_clone(const luci::CircleNode *node)
  {
    auto it = _clonecontext.find(node);
    if (it == _clonecontext.end())
      throw oops::UserExn("Invalid node in ConnectNode");
    return it->second;
  }

protected:
  luci::CloneContext &_clonecontext;
};

void ConnectNode::visit(const luci::CircleAdd *node)
{
  auto *cloned = loco::must_cast<luci::CircleAdd *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleConst *)
{
  // Nothing to do
}

void ConnectNode::visit(const luci::CircleDiv *node)
{
  auto *cloned = loco::must_cast<luci::CircleDiv *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleMean *node)
{
  auto *cloned = loco::must_cast<luci::CircleMean *>(find_clone(node));
  luci::CircleNode *in_i = loco::must_cast<luci::CircleNode *>(node->input());
  luci::CircleNode *in_r = loco::must_cast<luci::CircleNode *>(node->reduction_indices());
  cloned->input(find_clone(in_i));
  cloned->reduction_indices(find_clone(in_r));
}

void ConnectNode::visit(const luci::CircleMul *node)
{
  auto *cloned = loco::must_cast<luci::CircleMul *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CirclePow *node)
{
  auto *cloned = loco::must_cast<luci::CirclePow *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleRsqrt *node)
{
  auto *cloned = loco::must_cast<luci::CircleRsqrt *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  cloned->x(find_clone(in_x));
}

void ConnectNode::visit(const luci::CircleSqrt *node)
{
  auto *cloned = loco::must_cast<luci::CircleSqrt *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  cloned->x(find_clone(in_x));
}

void ConnectNode::visit(const luci::CircleSquaredDifference *node)
{
  auto *cloned = loco::must_cast<luci::CircleSquaredDifference *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

void ConnectNode::visit(const luci::CircleSub *node)
{
  auto *cloned = loco::must_cast<luci::CircleSub *>(find_clone(node));
  luci::CircleNode *in_x = loco::must_cast<luci::CircleNode *>(node->x());
  luci::CircleNode *in_y = loco::must_cast<luci::CircleNode *>(node->y());
  cloned->x(find_clone(in_x));
  cloned->y(find_clone(in_y));
}

} // namespace

namespace luci
{

void clone_connect(const CircleNode *node, CloneContext &clonecontext)
{
  ConnectNode cn(clonecontext);
  node->accept(&cn);
}

} // namespace luci
