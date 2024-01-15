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

#include "luci/Pass/CopyQuantParamPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Log.h>

namespace luci
{

namespace
{

struct SrcDst
{
  CircleNode *src = nullptr;
  CircleNode *dst = nullptr;
};

} // namespace

bool CopyQuantParamPass::run(loco::Graph *g)
{
  LOGGER(l);

  INFO(l) << "CopyQuantParamPass Start" << std::endl;

  if (_src_tensors.size() != _dst_tensors.size())
    throw std::runtime_error("The numbers of Source/Destination tensors do not match.");

  // Return src/dst CircleNodes
  auto get_src_dst = [&g](std::string src, std::string dst) {
    SrcDst src_dst;
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto const cnode = loco::must_cast<CircleNode *>(node);
      auto const name = cnode->name();
      if (name == src)
        src_dst.src = cnode;

      if (name == dst)
        src_dst.dst = cnode;
    }
    return src_dst;
  };

  for (uint32_t i = 0; i < _src_tensors.size(); i++)
  {
    auto &src = _src_tensors[i];
    auto &dst = _dst_tensors[i];

    auto nodes = get_src_dst(src, dst);
    if (not nodes.src)
      throw std::runtime_error("The tensor named " + src + " does not exist.");

    if (not nodes.dst)
      throw std::runtime_error("The tensor named " + dst + " does not exist.");

    copy_quantparam(nodes.src, nodes.dst);

    INFO(l) << "Quantparam of " << src << " is copied to " << dst << std::endl;
  }

  INFO(l) << "CopyQuantParamPass End" << std::endl;

  return false; // one time run
}

} // namespace luci
