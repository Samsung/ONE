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

#include "luci/Pass/PropagateQuantParamPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <iostream>

namespace
{

bool copy_qparam(luci::CircleQuantParam *src, luci::CircleQuantParam *dst)
{
  assert(src->scale.size() == dst->scale.size());
  assert(src->zerop.size() == dst->zerop.size());

  // src and dst have the same qparam
  if (std::equal(src->scale.begin(), src->scale.end(), dst->scale.begin()) &&
      std::equal(src->zerop.begin(), src->zerop.end(), dst->zerop.begin()) &&
      src->quantized_dimension == dst->quantized_dimension)
    return false;

  dst->scale.assign(src->scale.begin(), src->scale.end());
  dst->zerop.assign(src->zerop.begin(), src->zerop.end());
  dst->quantized_dimension = src->quantized_dimension;
  return true;
}

bool copy_qparam(luci::CircleNode *src, luci::CircleNode *dst)
{
  // Skip nodes that do not have quantparams
  auto src_qparam = src->quantparam();
  if (not src_qparam)
    return false;

  auto dst_qparam = dst->quantparam();
  if (not dst_qparam)
    return false;

  return copy_qparam(src_qparam, dst_qparam);
}

//  Visitor to propagate quantization parameters
struct PropagateQuantParam final : public luci::CircleNodeMutableVisitor<bool>
{
  PropagateQuantParam() = default;

  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleReshape *node)
  {
    auto input = node->tensor();
    if (loco::succs(input).size() != 1)
      return false;

    auto input_node = loco::must_cast<luci::CircleNode *>(input);
    return copy_qparam(input_node, node);
  }

  bool visit(luci::CircleTranspose *node)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->a());
    return copy_qparam(input_node, node);
  }

  // TODO : Add more Ops (e.g., layout-changing Ops)
};

} // namespace

namespace luci
{

bool PropagateQuantParamPass::run(loco::Graph *g)
{
  bool changed = false;
  LOGGER(l);
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    INFO(l) << "PropagateQuantParamPass visit node: " << circle_node->name() << std::endl;

    PropagateQuantParam pqp;
    if (circle_node->accept(&pqp))
      changed = true;
  }

  return changed;
}

} // namespace luci
