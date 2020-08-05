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
#include <iostream>
#include <luci/IR/CircleNodes.h>

namespace
{

bool fused_batch_norm_with_tconv(luci::CircleTransposeConv *tconv)
{
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

  // get scale of batchnorm
  auto scale = dynamic_cast<luci::CircleConst *>(mul->y());
  if (not scale)
    return false;

  // mul dim(0) == tconv filter channel dim
  if (filter->rank() != 4)
    return false;
  auto filter_channel_dim = filter->dim(3).value();
  if (scale->rank() != 1)
    return false;
  auto mul_dim = scale->dim(0).value();
  if (filter_channel_dim != mul_dim)
    return false;

  // filter weight = filter weight * mul
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
  // remove mul node
  replace(mul).with(tconv);
  // TODO add bias attribute to tconv
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
