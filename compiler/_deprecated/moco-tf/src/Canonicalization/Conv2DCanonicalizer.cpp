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

#include "Conv2DCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include "CodecHelper.h"

#include <moco/Log.h>

namespace
{
using plier::tf::DataLayout;

void set_filter_enc(loco::FilterEncode *filter_enc)
{
  auto enc = std::make_unique<loco::PermutingEncoder<loco::Domain::Filter>>();

  // In TensorFlow, conv2d filter is a 4-D tensor of following shape:
  // [filter_height, filter_width, in_channels, out_channels] -> HWIO (HWCN)
  enc->perm()->axis(loco::FilterAxis::Height) = 0;
  enc->perm()->axis(loco::FilterAxis::Width) = 1;
  enc->perm()->axis(loco::FilterAxis::Depth) = 2;
  enc->perm()->axis(loco::FilterAxis::Count) = 3;

  filter_enc->encoder(std::move(enc));
}

bool canonicalize_conv2d(loco::Graph *graph, moco::TFConv2D *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFCon2D node with Canonical FeatureEncode +
   *       FilterEncode + Conv2D + FeatureDecode
   *
   *       Before
   *                 A -- TFConv2D - C
   *                 B -/
   *
   *       After
   *                 A -- TFConv2D -
   *                 B -/
   *                 A -- FeatureEncode - Conv2D - FeatureDecode - C
   *                 B -- FilterEncode -/
   *
   *       Where
   *                 A : ifm of TFConv2D
   *                 B : ker of TFConv2D
   *                 C : a node that uses TFConv2D as an input
   *                 TFConv2D is disconnected from other nodes
   *                 A and B are drawn twice to simplify the diagram
   */

  auto data_layout = plier::tf::as_data_layout(node->data_layout());

  auto feature_enc = graph->nodes()->create<loco::FeatureEncode>();
  auto filter_enc = graph->nodes()->create<loco::FilterEncode>();
  auto conv2d = graph->nodes()->create<loco::Conv2D>();
  auto feature_dec = graph->nodes()->create<loco::FeatureDecode>();

  set_feature_enc(feature_enc, data_layout);
  set_filter_enc(filter_enc);
  set_feature_dec(feature_dec, data_layout);

  auto input_shape = moco::node_shape(node->input());
  assert(input_shape.domain() != loco::Domain::Unknown);

  auto ker_shape = moco::node_shape(node->filter());
  auto ker_tensor_shape = ker_shape.as<loco::TensorShape>(); // in HWIO

  auto node_stride = moco::stride_of(node->strides(), node->data_layout());
  auto node_window = moco::window_of(ker_tensor_shape, "HWIO");

  moco::Padding2DInference infer_padding2d;

  infer_padding2d.padding(node->padding());
  infer_padding2d.stride(node_stride);
  infer_padding2d.window(node_window);

  auto input_feature_shape = moco::as_feature_shape(input_shape, node->data_layout());
  auto input_plane_shape = moco::make_plane_shape(input_feature_shape);

  *conv2d->pad() = infer_padding2d(input_plane_shape);
  *conv2d->stride() = node_stride;

  // update graph
  auto node_A = node->input();
  auto node_B = node->filter();

  // update connections
  feature_enc->input(node_A);
  filter_enc->input(node_B);
  conv2d->ifm(feature_enc);
  conv2d->ker(filter_enc);
  feature_dec->input(conv2d);

  // replace old node
  replace(node).with(feature_dec);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool Conv2DCanonicalizer::transform(TFConv2D *node) const
{
  return canonicalize_conv2d(node->graph(), node);
}

} // namespace tf
} // namespace moco
