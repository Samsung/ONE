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

#include "luci/Pass/PropagateQParamBackwardPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

namespace
{

// Visitor to propagate quantization parameters backwards
struct PropagateQParamBackward final : public luci::CircleNodeMutableVisitor<void>
{
  PropagateQParamBackward(loco::DataType output) : _output_type(output) {}

private:
  loco::DataType _output_type;

  void visit(luci::CircleNode *) {}
};

} // namespace

namespace luci
{

bool PropagateQParamBackwardPass::run(loco::Graph *g)
{
  LOGGER(l);

  // We use post-order traversal as qparam is propagated backward
  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    INFO(l) << "PropagateQParamBackwardPass visit node: " << circle_node->name() << std::endl;

    PropagateQParamBackward pqb(_output_model_dtype);
    circle_node->accept(&pqb);
  }

  return false;
}

} // namespace luci
