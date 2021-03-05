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

#include "luci/Pass/ResolveCustomOpBatchMatMulPass.h"

#include "flatbuffers/flexbuffers.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool resolve_custom_op(luci::CircleCustom *cop)
{
  const std::string custom_code = cop->custom_code();
  const std::vector<uint8_t> custom_options = cop->custom_options();

  if (custom_code == "BatchMatMulV2")
  {
    auto name = cop->name();
    assert(name.length() > 0);

    auto batch_matmul = cop->graph()->nodes()->create<luci::CircleBatchMatMul>();
    // input
    batch_matmul->x(cop->inputs(0));
    batch_matmul->y(cop->inputs(1));
    // TODO find much better way of parsing custom_options
    // adj
    auto map = flexbuffers::GetRoot(custom_options).AsMap();
    batch_matmul->adj_x(map["adj_x"].AsBool());
    batch_matmul->adj_y(map["adj_y"].AsBool());
    batch_matmul->name(name + "/BatchMatMul");

    auto customOut = loco::succs(cop);
    assert(customOut.size() == 1);
    replace(*customOut.begin()).with(batch_matmul);

    return true;
  }

  return false;
}

} // namespace

namespace luci
{

/**
 *  BEFORE
 *         |             |
 *    [CircleNode]  [CircleNode]
 *          \           /
 *         [CircleCustom]("BatchMatMulV2")
 *               |
 *        [CircleCustomOut]
 *               |
 *          [CircleNode]
 *               |
 *
 *  AFTER
 *         |             |
 *    [CircleNode]  [CircleNode]
 *          \           /
 *       [CircleBatchMatMul]
 *               |
 *          [CircleNode]
 *               |
 */
bool ResolveCustomOpBatchMatMulPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    if (resolve_custom_op(cop))
      changed = true;
  }

  return changed;
}

} // namespace luci
