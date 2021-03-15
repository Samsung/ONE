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

#include "luci/Pass/FuseBatchNormWithTConvPass.h"

#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{
/**
 *  Fuse Mul-Add to TransposeConv if possible.
 *
 *  NOTE TF's BatchNormalization is converted to Mul and Add.
 *
 *  BEFORE
 *                     |   [CircleOutputExclude]
 *                     |   / [CircleConst]
 *                     |  / /
 *     [CircleTransposeConv]  [CircleConst]
 *                     |     /
 *                [CircleMul] [CircleConst]
 *                     |     /
 *                [CircleAdd]
 *                     |
 *
 *  AFTER
 *                     |                                          [CircleOutputExclude]
 *                     +-------------------------------------+   / [CircleConst]
 *                     |                                     |  / /
 *                     |                     [CircleTransposeConv]  [CircleConst]
 *                     |    [CircleConst]                    |     /
 *                     |   / [CircleConst]              [CircleMul] [CircleConst]
 *                     |  / /                                |     /
 *     [CircleTransposeConv]                            [CircleAdd]
 *                     |
 *              ([CircleRelu6])
 *                     |
 *
 * Note: CircleRelu6 is inserted if Add activation is ReLU6
 */
bool fused_batch_norm_with_tconv(luci::CircleAdd *add)
{
  assert(add != nullptr);

  // Find the pattern of CircleTransposeConv - CircleMul - CircleAdd
  luci::CircleConst *scale = nullptr;
  luci::CircleConst *shift = nullptr;
  luci::CircleTransposeConv *tconv = nullptr;
  luci::CircleMul *mul = nullptr;
  if (not luci::fill(&shift, &mul).with_commutative_args_of(add))
    return false;
  if (not luci::fill(&scale, &tconv).with_commutative_args_of(mul))
    return false;

  // check scale and shift constant attributes
  if (scale->rank() != 1)
    return false;
  if (shift->rank() != 1)
    return false;
  // check mul, add attributes
  if (mul->dtype() != loco::DataType::FLOAT32)
    return false;
  if (add->dtype() != loco::DataType::FLOAT32)
    return false;
  if (add->fusedActivationFunction() != luci::FusedActFunc::NONE &&
      add->fusedActivationFunction() != luci::FusedActFunc::RELU6)
    return false;

  // tconv bias should be not set
  if (not dynamic_cast<luci::CircleOutputExclude *>(tconv->bias()))
    return false;

  // get weight of tconv
  auto filter = dynamic_cast<luci::CircleConst *>(tconv->filter());
  if (not filter)
    return false;
  if (filter->dtype() != loco::DataType::FLOAT32)
    return false;
  if (filter->rank() != 4)
    return false;

  auto filter_out_chn = filter->dim(0).value();
  if (filter_out_chn != scale->dim(0).value())
    return false;
  if (filter_out_chn != shift->dim(0).value())
    return false;

  auto name = add->name();
  assert(name.length() > 0);

  loco::Graph *graph = add->graph();
  luci::CircleTransposeConv *fused_tconv = graph->nodes()->create<luci::CircleTransposeConv>();
  luci::CircleConst *fused_filter = graph->nodes()->create<luci::CircleConst>();
  luci::CircleConst *fused_bias = graph->nodes()->create<luci::CircleConst>();

  auto filter_height = filter->dim(1).value();
  auto filter_width = filter->dim(2).value();
  auto filter_in_chn = filter->dim(3).value();

  // Copy filter shape
  fused_filter->dtype(filter->dtype());
  fused_filter->size<loco::DataType::FLOAT32>(filter->size<loco::DataType::FLOAT32>());
  fused_filter->rank(4);
  fused_filter->dim(0).set(filter_out_chn);
  fused_filter->dim(1).set(filter_height);
  fused_filter->dim(2).set(filter_width);
  fused_filter->dim(3).set(filter_in_chn);
  fused_filter->shape_status(luci::ShapeStatus::VALID);
  fused_filter->name(name + "/TransposeConv/filter");

  // fused filter weight = filter weight * mul(scale) + add(shift)
  for (uint32_t c = 0; c < filter_out_chn; c++)
  {
    for (uint32_t h = 0; h < filter_height; h++)
    {
      for (uint32_t w = 0; w < filter_width; w++)
      {
        for (uint32_t b = 0; b < filter_in_chn; b++)
        {
          uint32_t offset = c * filter_height * filter_width * filter_in_chn +
                            h * filter_width * filter_in_chn + w * filter_in_chn + b;
          fused_filter->at<loco::DataType::FLOAT32>(offset) =
            filter->at<loco::DataType::FLOAT32>(offset) * scale->at<loco::DataType::FLOAT32>(c);
        }
      }
    }
  }

  // Copy fused_bias from shift
  fused_bias->dtype(shift->dtype());
  fused_bias->size<loco::DataType::FLOAT32>(shift->size<loco::DataType::FLOAT32>());
  fused_bias->rank(1);
  fused_bias->dim(0).set(filter_out_chn);
  fused_bias->shape_status(luci::ShapeStatus::VALID);
  for (uint32_t c = 0; c < filter_out_chn; ++c)
  {
    fused_bias->at<loco::DataType::FLOAT32>(c) = shift->at<loco::DataType::FLOAT32>(c);
  }
  fused_bias->name(name + "/TransposeConv/bias");

  // set new tconv properties
  fused_tconv->inputSizes(tconv->inputSizes());
  fused_tconv->filter(fused_filter);
  fused_tconv->outBackprop(tconv->outBackprop());
  fused_tconv->bias(fused_bias);
  fused_tconv->padding(tconv->padding());
  fused_tconv->stride()->h(tconv->stride()->h());
  fused_tconv->stride()->w(tconv->stride()->w());
  fused_tconv->name(name + "/TransposeConv");
  luci::add_origin(fused_tconv,
                   luci::composite_origin(
                     {luci::get_origin(add), luci::get_origin(mul), luci::get_origin(tconv)}));

  if (add->fusedActivationFunction() == luci::FusedActFunc::RELU6)
  {
    // separate relu op from add op
    auto relu = add->graph()->nodes()->create<luci::CircleRelu6>();
    relu->features(fused_tconv);
    relu->name(name + "/Relu6");
    luci::add_origin(relu, luci::get_origin(add));

    replace(add).with(relu);
  }
  else
  {
    replace(add).with(fused_tconv);
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
    if (auto add = dynamic_cast<luci::CircleAdd *>(node))
    {
      if (fused_batch_norm_with_tconv(add))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
