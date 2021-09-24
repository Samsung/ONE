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

#include "luci/Pass/FuseAddWithFullyConnectedPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
/**
 *  Fuse Add to FullyConnected if the added value is a channel-wise constant
 *
 *  BEFORE
 *                |
 *      [CircleFullyConnected]
 *                |
 *           [CircleAdd]
 *                |
 *
 *  AFTER
 *                |
 *       [CircleFullyConnected]   [CircleAdd] (dead)
 *                |
 *
 */
bool fuse_add_with_fc(luci::CircleFullyConnected *fc)
{
  if (not fc)
    return false;

  if (fc->dtype() != loco::DataType::FLOAT32)
    return false;

  if (fc->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
  if (not weights)
    return false;

  // get add node
  auto fc_output = loco::succs(fc);
  if (fc_output.size() != 1)
    return false;

  auto add = dynamic_cast<luci::CircleAdd *>(*fc_output.begin());
  if (not add)
    return false;
  if (add->dtype() != loco::DataType::FLOAT32)
    return false;

  // get addition
  auto addition = add->x() == fc ? dynamic_cast<luci::CircleConst *>(add->y())
                                 : dynamic_cast<luci::CircleConst *>(add->x());

  // Non-const addition
  if (not addition)
    return false;

  auto rank = addition->rank();
  // TODO Support scalar addition
  if (rank == 0)
    return false;

  // Check addition is channel-wise constant (e.g., shape = [c], [1, c] or [1, 1, 1, c])
  for (uint32_t i = 0; i < rank - 1; i++)
  {
    if (addition->dim(i).value() != 1)
      return false;
  }
  auto filter_channel_dim = weights->dim(0).value();
  if (addition->dim(rank - 1).value() != filter_channel_dim)
    return false;

  auto fused_bias = luci::clone(addition);

  // Add existing bias values
  if (auto const_bias = dynamic_cast<luci::CircleConst *>(fc->bias()))
  {
    assert(filter_channel_dim == fused_bias->size<loco::DataType::FLOAT32>());
    assert(const_bias->dtype() == loco::DataType::FLOAT32);

    for (uint32_t c = 0; c < filter_channel_dim; ++c)
      fused_bias->at<loco::DataType::FLOAT32>(c) += const_bias->at<loco::DataType::FLOAT32>(c);
  }

  fc->bias(fused_bias);
  fc->fusedActivationFunction(add->fusedActivationFunction());

  // set origin
  luci::add_origin(fc, luci::get_origin(add));

  replace(add).with(fc);

  return true;
}

} // namespace

namespace luci
{

bool FuseAddWithFullyConnectedPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;

    if (fuse_add_with_fc(fc))
      changed = true;
  }

  return changed;
}

} // namespace luci
