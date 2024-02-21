/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseAddWithConvPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
/**
 *  Fuse Add to Conv2D if possible.
 *
 *  BEFORE
 *                  |   [CircleConst]
 *                  |  / [CircleConst]
 *                  | / /
 *         [CircleConv2D] [CircleConst]
 *                  |    /
 *             [CircleAdd]
 *                  |
 *
 *  AFTER
 *                  |                  [CircleConst]
 *                  +--------------+  / [CircleConst]
 *                  |              | / /
 *                  |     [CircleConv2D] [CircleConst]
 *  [CircleConst]   |              |    /
 * [CircleConst] \  |         [CircleAdd]
 *              \ \ |
 *           [CircleConv2D]
 *                  |
 */
bool fused_add_with_conv(luci::CircleAdd *add)
{
  // find the pattern of CircleAdd(CircleConv2D, CircleConst)
  luci::CircleConst *shift = nullptr;
  luci::CircleConv2D *conv2d = nullptr;
  if (not luci::fill(&conv2d, &shift).with_commutative_args_of(add))
    return false;

  // check conditions for conv2d
  if (conv2d->rank() != 4)
    return false;
  if (conv2d->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;

  luci::CircleConst *filter = dynamic_cast<luci::CircleConst *>(conv2d->filter());
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(conv2d->bias());
  luci::CircleOutputExclude *biasex = dynamic_cast<luci::CircleOutputExclude *>(conv2d->bias());

  // filter should exist, bias should be const or none(output exclude)
  if (filter == nullptr || (bias == nullptr && biasex == nullptr))
    return false;
  if (filter->rank() != 4)
    return false;
  if (filter->dtype() != shift->dtype())
    return false;
  // TODO support more data type
  if (filter->dtype() != loco::DataType::FLOAT32)
    return false;

  // filter is OHWI
  uint32_t out_channel = filter->dim(0).value();

  // shape of shift should be [1, 1, 1, out_channel] or [out_channel]
  if (shift->rank() == 4)
  {
    for (uint32_t i = 0; i < 3; ++i)
      if (shift->dim(i).value() != 1)
        return false;
    if (shift->dim(3).value() != out_channel)
      return false;
  }
  else if (shift->rank() == 1)
  {
    if (shift->dim(0).value() != out_channel)
      return false;
  }
  else
    return false;

  auto conv2d_name = conv2d->name();
  auto shift_name = shift->name();
  assert(conv2d_name.length() > 0);
  assert(shift_name.length() > 0);
  auto bias_name = (bias ? bias->name() : conv2d_name) + ";" + shift_name;

  luci::CircleConv2D *fused_conv2d = add->graph()->nodes()->create<luci::CircleConv2D>();
  luci::CircleConst *fused_bias = add->graph()->nodes()->create<luci::CircleConst>();

  fused_bias->dtype(conv2d->dtype());
  fused_bias->rank(1);
  fused_bias->dim(0).set(out_channel);
  fused_bias->shape_status(luci::ShapeStatus::VALID);
  fused_bias->name(bias_name);
  fused_bias->size<loco::DataType::FLOAT32>(out_channel);
  // fuse shift to bias
  for (uint32_t b = 0; b < out_channel; ++b)
  {
    auto bias_val = shift->at<loco::DataType::FLOAT32>(b);
    if (bias)
      bias_val += bias->at<loco::DataType::FLOAT32>(b);
    fused_bias->at<loco::DataType::FLOAT32>(b) = bias_val;
  }

  // Set attributes of fused_conv2d
  fused_conv2d->input(conv2d->input());
  fused_conv2d->filter(conv2d->filter());
  fused_conv2d->bias(fused_bias);
  fused_conv2d->fusedActivationFunction(add->fusedActivationFunction());
  fused_conv2d->padding(conv2d->padding());
  fused_conv2d->stride()->h(conv2d->stride()->h());
  fused_conv2d->stride()->w(conv2d->stride()->w());
  fused_conv2d->dilation()->h(conv2d->dilation()->h());
  fused_conv2d->dilation()->w(conv2d->dilation()->w());
  fused_conv2d->name(conv2d_name);
  luci::add_origin(fused_conv2d,
                   luci::composite_origin({luci::get_origin(add), luci::get_origin(conv2d)}));

  replace(add).with(fused_conv2d);

  return true;
}

} // namespace

namespace luci
{

bool FuseAddWithConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
    {
      if (fused_add_with_conv(add))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
