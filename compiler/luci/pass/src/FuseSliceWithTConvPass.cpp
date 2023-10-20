/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseSliceWithTConvPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/CircleNodeClone.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

#define RETURN_FALSE_UNLESS(cond) \
  if (not(cond))                  \
    return false;

inline int32_t compute_input_size(luci::Padding padding, int32_t image_size, int32_t filter_size,
                                  int32_t stride)
{
  switch (padding)
  {
    case luci::Padding::SAME:
      return (image_size + stride - 1) / stride;
    case luci::Padding::VALID:
      return (image_size + stride - filter_size) / stride;
    default:
      throw std::runtime_error("Unsupported padding");
  }
}

inline int32_t extract_pad_value(int32_t stride, int32_t in_size, int32_t filter_size,
                                 int32_t out_size)
{
  const int32_t padding = ((in_size - 1) * stride + filter_size - out_size) / 2;
  return padding > 0 ? padding : 0;
}

inline uint32_t cal_offset(const luci::CircleConst *node, const uint32_t *indices)
{
  return indices[0] * node->dim(1).value() * node->dim(2).value() * node->dim(3).value() +
         indices[1] * node->dim(2).value() * node->dim(3).value() +
         indices[2] * node->dim(3).value() + indices[3];
}

luci::Padding compute_padding(const luci::CircleTransposeConv *tconv, int32_t out_height,
                              int32_t out_width, int32_t pad_top, int32_t pad_left)
{
  auto const filter = dynamic_cast<luci::CircleConst *>(tconv->filter());
  if (!filter)
    return luci::Padding::UNDEFINED;

  auto tconv_shape = dynamic_cast<luci::CircleConst *>(tconv->inputSizes());
  if (!tconv_shape)
    return luci::Padding::UNDEFINED;

  luci::Padding padding = luci::Padding::UNDEFINED;
  std::initializer_list<luci::Padding> paddings_to_check = {luci::Padding::VALID,
                                                            luci::Padding::SAME};

  auto const filter_height = filter->dim(1).value();
  auto const filter_width = filter->dim(2).value();
  auto const stride_height = tconv->stride()->h();
  auto const stride_width = tconv->stride()->w();

  for (auto padding_to_check : paddings_to_check)
  {
    auto const in_height =
      compute_input_size(padding_to_check, out_height, filter_height, stride_height);
    auto const pad_top_virtual =
      extract_pad_value(stride_height, in_height, filter_height, out_height);
    if (pad_top_virtual != pad_top)
      continue;

    auto const in_width =
      compute_input_size(padding_to_check, out_width, filter_width, stride_width);
    auto const pad_left_virtual =
      extract_pad_value(stride_width, in_width, filter_width, out_width);
    if (pad_left_virtual == pad_left)
    {
      padding = padding_to_check; // correct padding is found
      break;
    }
  }

  return padding;
}

/**
 *  Fuse Slice with CircleTransposeConv if possible
 *
 *  NOTE: In case predecessor of slice is tconv, we can try to merge slice with tconv,
 *  because spatial slice is reduction so as padding for tconv,
 *  while channels slice reduction can be directly modeled in tconv.
 *  For now there is no option to set explicitely pad values for
 *  CircleTransposeConv. Only using VALID/SAME and output shape is the only way
 *  to set pad values. That is why not all numerical values of pad are legal for such
 *  transform.
 *
 *  BEFORE
 *                    |
 *           [CircleTransposeConv]
 *                    |
 *               [CircleSlice]
 *                    |
 *
 *  AFTER
 *                    |
 *            [CircleTransposeConv] (with m.b. changed padding, output shape, and filter/bias)
 *                    |
 *
 */

bool fuse_slice_with_tconv(luci::CircleSlice *slice)
{
  // NOTE: assume NHWC layout
  auto tconv = dynamic_cast<luci::CircleTransposeConv *>(slice->input());
  RETURN_FALSE_UNLESS(tconv != nullptr);

  // offset
  auto begin = dynamic_cast<luci::CircleConst *>(slice->begin());
  // sanity check
  RETURN_FALSE_UNLESS(begin != nullptr && begin->dtype() == loco::DataType::S32 &&
                      begin->rank() == 1);

  // output shape
  auto out_shape = dynamic_cast<luci::CircleConst *>(slice->size());
  // sanity check
  RETURN_FALSE_UNLESS(out_shape != nullptr && out_shape->dtype() == loco::DataType::S32 &&
                      out_shape->rank() == 1);

  // output shape of tconv
  auto tconv_shape = dynamic_cast<luci::CircleConst *>(tconv->inputSizes());
  // sanity check
  RETURN_FALSE_UNLESS(tconv_shape != nullptr && tconv_shape->dtype() == loco::DataType::S32 &&
                      tconv_shape->rank() == 1);

  // no update if batch dimension is processed in slice
  RETURN_FALSE_UNLESS(begin->at<loco::DataType::S32>(0) == 0 &&
                      out_shape->at<loco::DataType::S32>(0) ==
                        tconv_shape->at<loco::DataType::S32>(0));

  // filter
  auto const tconv_filter = dynamic_cast<luci::CircleConst *>(tconv->filter());
  // sanity check
  RETURN_FALSE_UNLESS(tconv_filter != nullptr && tconv_filter->rank() == 4 &&
                      tconv_filter->dtype() == loco::DataType::FLOAT32);

  // bias
  auto const tconv_bias = dynamic_cast<luci::CircleConst *>(tconv->bias());
  // Only support const bias
  // TODO Support non-const bias
  RETURN_FALSE_UNLESS(tconv_bias != nullptr && tconv_bias->rank() == 1 &&
                      tconv_bias->dtype() == loco::DataType::FLOAT32);

  auto const out_height = out_shape->at<loco::DataType::S32>(1);
  auto const out_width = out_shape->at<loco::DataType::S32>(2);

  auto const pad_top = begin->at<loco::DataType::S32>(1);
  auto const pad_left = begin->at<loco::DataType::S32>(2);

  // As there is no option to set numerical values of pad explicitly for CircleTransposeConv
  // we need to be sure that interpretation of PADDING + OUTPUT_SHAPE will produce
  // the pad values, defined by slice. If possible compute_padding will return correct
  // padding value, otherwise it will return UNDEFINED
  auto const padding = compute_padding(tconv, out_height, out_width, pad_top, pad_left);
  if (padding == luci::Padding::UNDEFINED)
    return false; // impossible to fuse

  auto const out_channels = out_shape->at<loco::DataType::S32>(3);
  // update filter and bias in case it's needed
  loco::Node *fused_filter = tconv->filter();
  loco::Node *fused_bias = tconv->bias();
  // Channel-direction slice
  // Corresponding weights/bias of TConv is sliced.
  if (begin->at<loco::DataType::S32>(3) != 0 ||
      out_channels != tconv_shape->at<loco::DataType::S32>(3))
  {
    // fused filter
    auto const in_channels = tconv_filter->dim(3).value();

    luci::CircleConst *fused_tconv_filter = luci::clone(tconv_filter);
    fused_tconv_filter->dim(0).set(out_channels); // out_channels
    // update size due to channels change
    fused_tconv_filter->size<loco::DataType::FLOAT32>(out_channels * tconv_filter->dim(1).value() *
                                                      tconv_filter->dim(2).value() * in_channels);
    auto const ch_offset = begin->at<loco::DataType::S32>(3);
    // set reduced filter values
    for (uint32_t out_chan = 0; out_chan < fused_tconv_filter->dim(0).value(); out_chan++)
    {
      for (uint32_t out_height = 0; out_height < fused_tconv_filter->dim(1).value(); out_height++)
      {
        for (uint32_t out_width = 0; out_width < fused_tconv_filter->dim(2).value(); out_width++)
        {
          for (uint32_t in_chan = 0; in_chan < fused_tconv_filter->dim(3).value(); in_chan++)
          {
            uint32_t indices[4] = {out_chan, out_height, out_width, in_chan};
            uint32_t old_indices[4] = {out_chan + ch_offset, out_height, out_width, in_chan};
            auto const data =
              tconv_filter->at<loco::DataType::FLOAT32>(cal_offset(tconv_filter, old_indices));
            fused_tconv_filter->at<loco::DataType::FLOAT32>(
              cal_offset(fused_tconv_filter, indices)) = data;
          }
        }
      }
    }
    fused_tconv_filter->name(tconv_filter->name() + "/FusedSlice");
    luci::add_origin(fused_tconv_filter, luci::get_origin(tconv_shape));
    fused_filter = fused_tconv_filter;

    // fused bias
    luci::CircleConst *fused_tconv_bias = luci::clone(tconv_bias);
    fused_tconv_bias->size<loco::DataType::FLOAT32>(out_channels);
    fused_tconv_bias->dim(0).set(out_channels); // out_channels
    // set reduced bias values
    for (int32_t c = 0; c < out_channels; c++)
    {
      auto const data = tconv_bias->at<loco::DataType::FLOAT32>(c + ch_offset);
      fused_tconv_bias->at<loco::DataType::FLOAT32>(c) = data;
    }

    fused_tconv_bias->name(tconv_bias->name() + "/FusedSlice");
    luci::add_origin(fused_tconv_bias, luci::get_origin(tconv_bias));
    fused_bias = fused_tconv_bias;
  }

  auto *fused_tconv_shape = luci::clone(tconv_shape);
  // spatial dimensions
  fused_tconv_shape->at<loco::DataType::S32>(1) = out_height;
  fused_tconv_shape->at<loco::DataType::S32>(2) = out_width;
  // channels
  fused_tconv_shape->at<loco::DataType::S32>(3) = out_channels;
  fused_tconv_shape->name(tconv_shape->name() + "/FusedSlice");
  luci::add_origin(fused_tconv_shape, luci::get_origin(tconv_shape));

  // Configure new CircleTransposeConv operation.
  auto *fused_tconv =
    loco::must_cast<luci::CircleTransposeConv *>(luci::clone_node(tconv, slice->graph()));
  fused_tconv->inputSizes(fused_tconv_shape);
  fused_tconv->outBackprop(tconv->outBackprop());
  fused_tconv->filter(fused_filter);
  fused_tconv->bias(fused_bias);
  fused_tconv->padding(padding);
  fused_tconv->name(tconv->name() + "/FusedSlice");
  luci::add_origin(fused_tconv,
                   luci::composite_origin({luci::get_origin(tconv), luci::get_origin(slice)}));

  // Replace old slice operation with new fused_tconv with merged pad values
  replace(slice).with(fused_tconv);

  return true;
}

} // namespace

namespace luci
{

bool FuseSliceWithTConvPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto slice = dynamic_cast<luci::CircleSlice *>(node);
    if (not slice)
      continue;

    if (fuse_slice_with_tconv(slice))
      changed = true;
  }

  return changed;
}

} // namespace luci
