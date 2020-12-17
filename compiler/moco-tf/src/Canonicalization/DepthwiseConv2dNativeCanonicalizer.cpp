/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DepthwiseConv2dNativeCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include "CodecHelper.h"

#include <moco/Log.h>

namespace
{

using plier::tf::DataLayout;

void set_filter_enc(loco::DepthwiseFilterEncode *filter_enc)
{
  auto enc = stdex::make_unique<loco::PermutingEncoder<loco::Domain::DepthwiseFilter>>();

  // In TensorFlow, depthwiseconv2dnative filter is a 4-D tensor of following shape:
  // [filter_height, filter_width, in_channels, channel_multiplier] -> HWCM
  enc->perm()->axis(loco::DepthwiseFilterAxis::Height) = 0;
  enc->perm()->axis(loco::DepthwiseFilterAxis::Width) = 1;
  enc->perm()->axis(loco::DepthwiseFilterAxis::Depth) = 2;
  enc->perm()->axis(loco::DepthwiseFilterAxis::Multiplier) = 3;

  filter_enc->encoder(std::move(enc));
}

bool canonicalize_depthwiseconv2dnative(loco::Graph *graph, moco::TFDepthwiseConv2dNative *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFDepthwiseConv2dNative node with Canonical FeatureEncode +
   *       DepthwiseFilterEncode + DepthwiseConv2D + FeatureDecode
   *
   *       Before
   *              A -+- TFDepthwiseConv2dNative - C
   *                 |
   *              B -+
   *
   *       After
   *
   *            A -+ FeatureEncode ----------------+- DepthwiseConv2D - FeatureDecode - C
   *               |                               |
   *               +-(TFDepthwiseConv2dNative)     |
   *               |                               |
   *            B -+ DepthwiseFilterEncode --------+
   *
   *       Where
   *                 A : ifm of TFDepthwiseConv2dNative
   *                 B : ker of TFDepthwiseConv2dNative
   *                 C : a node that uses TFDepthwiseConv2dNative as an input
   *                 TFDepthwiseConv2dNative is disconnected from other nodes
   */

  INFO(l) << "TFNodeCanonicalize TFDepthwiseConv2dNative begin";

  auto data_layout = plier::tf::as_data_layout(node->data_layout());

  auto feature_enc = graph->nodes()->create<loco::FeatureEncode>();
  auto filter_enc = graph->nodes()->create<loco::DepthwiseFilterEncode>();
  auto depthwiseconv2d = graph->nodes()->create<loco::DepthwiseConv2D>();
  auto feature_dec = graph->nodes()->create<loco::FeatureDecode>();

  set_feature_enc(feature_enc, data_layout);
  set_filter_enc(filter_enc);
  set_feature_dec(feature_dec, data_layout);

  // Calculate Pad and Stride from inference
  auto input_shape = moco::node_shape(node->input());
  auto ker_shape = moco::node_shape(node->filter());
  auto ker_tensor_shape = ker_shape.as<loco::TensorShape>();
  auto node_stride = moco::stride_of(node->strides(), node->data_layout());
  auto node_window = moco::window_of(ker_tensor_shape, "HWCM");

  moco::Padding2DInference infer_padding2d;

  infer_padding2d.padding(node->padding());
  infer_padding2d.stride(node_stride);
  infer_padding2d.window(node_window);

  auto input_feature_shape = moco::as_feature_shape(input_shape, node->data_layout());
  auto input_plane_shape = moco::make_plane_shape(input_feature_shape);

  *depthwiseconv2d->pad() = infer_padding2d(input_plane_shape);
  *depthwiseconv2d->stride() = node_stride;

  // update graph
  auto node_A = node->input();
  auto node_B = node->filter();

  // update connections
  feature_enc->input(node_A);
  filter_enc->input(node_B);
  depthwiseconv2d->ifm(feature_enc);
  depthwiseconv2d->ker(filter_enc);
  feature_dec->input(depthwiseconv2d);

  // replace and disconnect old node
  replace(node).with(feature_dec);

  INFO(l) << "TFNodeCanonicalize TFDepthwiseConv2dNative done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool DepthwiseConv2dNativeCanonicalizer::transform(TFDepthwiseConv2dNative *node) const
{
  return canonicalize_depthwiseconv2dnative(node->graph(), node);
}

} // namespace tf
} // namespace moco
