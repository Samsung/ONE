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

#include "luci/Pass/FuseBatchNormWithDwConvPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>

namespace
{
/**
 *  Fuse Mul-Add to DepthwiseConv2D if possible.
 *
 *  NOTE TF's BatchNormalization is converted to Mul and Add.
 *
 *  BEFORE
 *                     |   [CircleConst]
 *                     |   / [CircleConst]
 *                     |  / /
 *    [CircleDepthwiseConv2D] [CircleConst]
 *                     |     /
 *                [CircleMul] [CircleConst]
 *                     |     /
 *                [CircleAdd]
 *                     |
 *
 *  AFTER
 *                     |                                          [CircleConst]
 *                     +-------------------------------------+   / [CircleConst]
 *                     |                                     |  / /
 *                     |                    [CircleDepthwiseConv2D] [CircleConst]
 *                     |    [CircleConst]                    |     /
 *                     |   / [CircleConst]              [CircleMul] [CircleConst]
 *                     |  / /                                |     /
 *    [CircleDepthwiseConv2D]                           [CircleAdd]
 *                     |
 *
 */

/**
 * @brief Check shape is [x] or [1, 1, 1, x]
 */
bool is_scale_shift_shape(luci::CircleConst *node)
{
  auto rank = node->rank();
  if (rank != 1 && rank != 4)
    return false;
  for (uint32_t r = 0; r < rank - 1; ++r)
  {
    if (node->dim(r).value() != 1)
      return false;
  }
  return true;
}

bool fused_batch_norm_with_dwconv(luci::CircleAdd *add)
{
  assert(add != nullptr);

  // Find the pattern of CircleDepthwiseConv2D - CircleMul - CircleAdd
  luci::CircleConst *scale = nullptr;
  luci::CircleConst *shift = nullptr;
  luci::CircleDepthwiseConv2D *dwconv = nullptr;
  luci::CircleMul *mul = nullptr;
  if (not luci::fill(&shift, &mul).with_commutative_args_of(add))
    return false;
  if (not luci::fill(&scale, &dwconv).with_commutative_args_of(mul))
    return false;

  // check scale and shift constant attributes
  // scale and shift can be [x] or [1, 1, 1, x]
  if (not is_scale_shift_shape(scale))
    return false;
  if (not is_scale_shift_shape(shift))
    return false;

  // check mul, add attributes
  if (mul->dtype() != loco::DataType::FLOAT32)
    return false;
  if (mul->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;
  if (add->dtype() != loco::DataType::FLOAT32)
    return false;
  // TODO support more Activations
  if (add->fusedActivationFunction() != luci::FusedActFunc::NONE &&
      add->fusedActivationFunction() != luci::FusedActFunc::RELU6)
    return false;

  // get weight of dwconv
  auto filter = dynamic_cast<luci::CircleConst *>(dwconv->filter());
  if (not filter)
    return false;
  if (filter->dtype() != loco::DataType::FLOAT32)
    return false;
  if (filter->rank() != 4)
    return false;

  // check attributes of dwconv
  if (dwconv->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return false;
  if (dwconv->depthMultiplier() < 0) // can this happen?
    return false;

  // get bias of dwconv
  auto bias = dynamic_cast<luci::CircleConst *>(dwconv->bias());
  if (not bias)
    return false;
  if (bias->dtype() != loco::DataType::FLOAT32)
    return false;
  if (bias->rank() != 1)
    return false;

  // filter represents as [1, H, W, C*M] where M is multiplier.
  auto filter_out_chn = filter->dim(3).value();
  auto multiplier = static_cast<uint32_t>(dwconv->depthMultiplier());
  auto srank = scale->rank(); // as rank can be 1 or 4
  if (filter_out_chn != scale->dim(srank - 1).value() * multiplier)
    return false;
  srank = shift->rank();
  if (filter_out_chn != shift->dim(srank - 1).value() * multiplier)
    return false;
  auto channel = filter_out_chn / multiplier;

  auto name = add->name();
  assert(name.length() > 0);

  loco::Graph *graph = add->graph();
  luci::CircleDepthwiseConv2D *fused_dwconv = graph->nodes()->create<luci::CircleDepthwiseConv2D>();
  luci::CircleConst *fused_filter = graph->nodes()->create<luci::CircleConst>();
  luci::CircleConst *fused_bias = graph->nodes()->create<luci::CircleConst>();

  auto filter_in_chn = filter->dim(0).value();
  auto filter_height = filter->dim(1).value();
  auto filter_width = filter->dim(2).value();
  assert(filter_in_chn == 1);

  // Copy filter shape
  fused_filter->dtype(filter->dtype());
  fused_filter->size<loco::DataType::FLOAT32>(filter->size<loco::DataType::FLOAT32>());
  fused_filter->rank(4);
  fused_filter->dim(0).set(filter_in_chn);
  fused_filter->dim(1).set(filter_height);
  fused_filter->dim(2).set(filter_width);
  fused_filter->dim(3).set(filter_out_chn);
  fused_filter->shape_status(luci::ShapeStatus::VALID);
  fused_filter->name(name + "/DepthwiseConv2D/filter");

  // fused filter weight = filter weight * mul(scale) + add(shift)
  for (uint32_t b = 0; b < filter_in_chn; b++)
  {
    for (uint32_t h = 0; h < filter_height; h++)
    {
      for (uint32_t w = 0; w < filter_width; w++)
      {
        for (uint32_t c = 0; c < filter_out_chn; c++)
        {
          uint32_t offset = b * filter_height * filter_width * filter_out_chn +
                            h * filter_width * filter_out_chn + w * filter_out_chn + c;
          uint32_t chn = c / multiplier;
          fused_filter->at<loco::DataType::FLOAT32>(offset) =
            filter->at<loco::DataType::FLOAT32>(offset) * scale->at<loco::DataType::FLOAT32>(chn);
        }
      }
    }
  }

  // Fuse bias with scale and shift
  fused_bias->dtype(shift->dtype());
  fused_bias->size<loco::DataType::FLOAT32>(shift->size<loco::DataType::FLOAT32>());
  fused_bias->rank(1);
  fused_bias->dim(0).set(channel);
  fused_bias->shape_status(luci::ShapeStatus::VALID);
  for (uint32_t c = 0; c < channel; ++c)
  {
    fused_bias->at<loco::DataType::FLOAT32>(c) =
      bias->at<loco::DataType::FLOAT32>(c) * scale->at<loco::DataType::FLOAT32>(c) +
      shift->at<loco::DataType::FLOAT32>(c);
  }
  fused_bias->name(name + "/DepthwiseConv2D/bias");

  // set new tconv properties
  fused_dwconv->input(dwconv->input());
  fused_dwconv->filter(fused_filter);
  fused_dwconv->bias(fused_bias);
  fused_dwconv->fusedActivationFunction(add->fusedActivationFunction());
  fused_dwconv->padding(dwconv->padding());
  fused_dwconv->stride()->h(dwconv->stride()->h());
  fused_dwconv->stride()->w(dwconv->stride()->w());
  fused_dwconv->depthMultiplier(dwconv->depthMultiplier());
  fused_dwconv->dilation()->h(dwconv->dilation()->h());
  fused_dwconv->dilation()->w(dwconv->dilation()->w());
  fused_dwconv->name(name + "/DepthwiseConv2D");

  replace(add).with(fused_dwconv);

  return true;
}

} // namespace

namespace luci
{

bool FuseBatchNormWithDwConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
    {
      if (fused_batch_norm_with_dwconv(add))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
