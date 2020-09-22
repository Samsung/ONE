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

#include "luci/Pass/FuseAddWithTConvPass.h"

#include <luci/IR/CircleNodes.h>

namespace
{
/**
 *  Fuse add to TCONV if possible
 *
 *  BEFORE
 *
 *         [CircleTransposeConv]
 *                  |
 *                [add]
 *  AFTER
 *
 *         [CircleTransposeConv]
 */
bool fuse_add_with_tconv(luci::CircleTransposeConv *tconv)
{
  // check whether it has bias or not. This optimization works only if it doesn't.
  auto bias = dynamic_cast<luci::CircleOutputExclude *>(tconv->bias());
  if (not bias)
    return false;

  // get weight of tconv
  auto filter = dynamic_cast<luci::CircleConst *>(tconv->filter());
  if (not filter)
    return false;
  if (filter->dtype() != loco::DataType::FLOAT32)
    return false;

  // get add node
  auto tconv_output = loco::succs(tconv);
  assert(tconv_output.size() == 1);
  auto add = dynamic_cast<luci::CircleAdd *>(*tconv_output.begin());
  if (not add)
    return false;
  if (add->dtype() != loco::DataType::FLOAT32)
    return false;
  if (add->fusedActivationFunction() != luci::FusedActFunc::NONE &&
      add->fusedActivationFunction() != luci::FusedActFunc::RELU6)
    return false;

  // get addition
  luci::CircleConst *addition = nullptr;
  if (add->x() == tconv)
    addition = dynamic_cast<luci::CircleConst *>(add->y());
  else
    addition = dynamic_cast<luci::CircleConst *>(add->x());

  if (not addition)
    return false;

  // addition dim(0) == tconv filter channel dim
  if (addition->rank() != 1)
    return false;
  auto addition_dim = addition->dim(0).value();
  auto filter_channel_dim = filter->dim(0).value();
  if (filter_channel_dim != addition_dim)
    return false;

  // fuse addition with transposed conv
  tconv->bias(addition);

  if (add->fusedActivationFunction() == luci::FusedActFunc::RELU6)
  {
    // separate relu op from add op
    auto relu = add->graph()->nodes()->create<luci::CircleRelu6>();
    relu->features(tconv);

    // remove add node
    replace(add).with(relu);
  }
  else
  {
    replace(add).with(tconv);
  }

  return true;
}

} // namespace

namespace luci
{

bool FuseAddWithTConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto tconv = dynamic_cast<luci::CircleTransposeConv *>(node);
    if (not tconv)
      continue;

    if (fuse_add_with_tconv(tconv))
      changed = true;
  }

  return changed;
}

} // namespace luci
