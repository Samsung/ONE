/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ForceQuantParamPass.h"
#include "luci/Profile/CircleNodeID.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Log.h>

namespace luci
{

namespace
{

void set_qparam(luci::CircleNode *node, float scale, int64_t zp)
{
  assert(node); // FIX_CALLER_UNLESS

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(scale);
  quantparam->zerop.push_back(zp);

  node->quantparam(std::move(quantparam));
}

} // namespace

bool ForceQuantParamPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "ForceQuantParamPass Start" << std::endl;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto const cnode = loco::must_cast<CircleNode *>(node);
    auto const name = cnode->name();
    auto target = std::find(_tensors.begin(), _tensors.end(), name);
    if (target == _tensors.end())
      continue;

    auto index = target - _tensors.begin();
    auto scale = _scales[index];
    auto zp = _zerops[index];
    set_qparam(cnode, scale, zp);

    _tensors.erase(_tensors.begin() + index);
    _scales.erase(_scales.begin() + index);
    _zerops.erase(_zerops.begin() + index);
  }

  if (_tensors.size() > 0)
  {
    std::string msg;
    for (auto const &t : _tensors)
      msg += "Tensor does not exist: " + t + ".\n";
    msg += "Please check tensor name.\n";
    throw std::runtime_error(msg);
  }

  INFO(l) << "ForceQuantParamPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
