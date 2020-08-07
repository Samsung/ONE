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

#include "luci/Pass/FuseBatchNormWithTConv.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool fused_batch_norm_with_tconv(luci::CircleTransposeConv *tconv)
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

  // get mul node
  auto tconv_output = loco::succs(tconv);
  assert(tconv_output.size() == 1);
  auto mul = dynamic_cast<luci::CircleMul *>(*tconv_output.begin());
  if (not mul)
    return false;
  if (mul->dtype() != loco::DataType::FLOAT32)
    return false;

  // get add node
  auto mul_output = loco::succs(mul);
  assert(mul_output.size() == 1);
  auto add = dynamic_cast<luci::CircleAdd *>(*mul_output.begin());
  if (not add)
    return false;
  if (add->dtype() != loco::DataType::FLOAT32)
    return false;
  if (add->fusedActivationFunction() != luci::FusedActFunc::NONE &&
      add->fusedActivationFunction() != luci::FusedActFunc::RELU6)
    return false;

  // get scale of batchnorm
  auto scale = dynamic_cast<luci::CircleConst *>(mul->y());
  if (not scale)
    return false;

  // scale dim(0) == tconv filter channel dim
  if (filter->rank() != 4)
    return false;
  auto filter_channel_dim = filter->dim(3).value();
  if (scale->rank() != 1)
    return false;
  auto scale_dim = scale->dim(0).value();
  if (filter_channel_dim != scale_dim)
    return false;

  // get shift of batchnorm
  auto shift = dynamic_cast<luci::CircleConst *>(add->y());
  if (not shift)
    return false;

  // shift dim(0) == tconv filter channel dim
  if (shift->rank() != 1)
    return false;
  auto shift_dim = shift->dim(0).value();
  if (filter_channel_dim != shift_dim)
    return false;

  // filter weight = filter weight * mul(scale) + add(shift)
  uint32_t filter_batch_dim = filter->dim(0).value();
  uint32_t filter_height_dim = filter->dim(1).value();
  uint32_t filter_width_dim = filter->dim(2).value();
  for (uint32_t c = 0; c < filter_channel_dim; c++)
  {
    for (uint32_t n = 0; n < filter_batch_dim; n++)
    {
      for (uint32_t h = 0; h < filter_height_dim; h++)
      {
        for (uint32_t w = 0; w < filter_width_dim; w++)
        {
          uint32_t offset = n * filter_height_dim * filter_width_dim * filter_channel_dim +
                            h * filter_width_dim * filter_channel_dim + w * filter_channel_dim + c;
          filter->at<loco::DataType::FLOAT32>(offset) *= scale->at<loco::DataType::FLOAT32>(c);
        }
      }
    }
  }

  // fuse shift with transposed conv
  tconv->bias(shift);

  if (add->fusedActivationFunction() == luci::FusedActFunc::RELU6)
  {
    // separate relu op from add op
    auto relu = add->graph()->nodes()->create<luci::CircleRelu6>();
    relu->features(tconv);

    // remove mul node
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

bool FuseBatchNormWithTConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto tconv = dynamic_cast<luci::CircleTransposeConv *>(node);
    if (not tconv)
      continue;

    changed |= fused_batch_norm_with_tconv(tconv);
  }

  return changed;
}

} // namespace luci
