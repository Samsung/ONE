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

#include "AvgPoolCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include "CodecHelper.h"

#include <loco/IR/NodeShape.h>

#include <moco/Log.h>

namespace
{

bool canonicalize_avgpool2d(loco::Graph *graph, moco::TFAvgPool *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFAvgPool node with Canonical FeatureEncode +
   *       AvgPool2D + FeatureDecode
   *
   *       Before
   *                 A -- TFAvgPool -- C
   *
   *       After
   *                    +- TFAvgPool --
   *                    |
   *                 A -+- FeatureEncode -- AvgPool2D -- FeatureDecode -- C
   *
   *       Where
   *                 A : value of TFAvgPool
   *                 C : a node that uses TFAvgPool as an input
   *                 TFAvgPool is disconnected from other nodes
   */

  auto data_layout = plier::tf::as_data_layout(node->data_layout());

  auto feature_enc = graph->nodes()->create<loco::FeatureEncode>();
  auto avgPool2d_node = graph->nodes()->create<loco::AvgPool2D>();
  auto feature_dec = graph->nodes()->create<loco::FeatureDecode>();

  set_feature_enc(feature_enc, data_layout);
  set_feature_dec(feature_dec, data_layout);

  avgPool2d_node->convention(loco::AvgPool2D::Convention::Valid);

  auto value_shape = moco::node_shape(node->value());
  assert(value_shape.domain() != loco::Domain::Unknown);

  auto node_stride = moco::stride_of(node->strides(), node->data_layout());
  auto node_window = moco::window_of(node->ksize(), node->data_layout());

  moco::Padding2DInference infer_padding2d;

  infer_padding2d.padding(node->padding());
  infer_padding2d.stride(node_stride);
  infer_padding2d.window(node_window);

  auto input_feature_shape = moco::as_feature_shape(value_shape, node->data_layout());
  auto input_plane_shape = moco::make_plane_shape(input_feature_shape);

  *avgPool2d_node->pad() = infer_padding2d(input_plane_shape);
  *avgPool2d_node->stride() = node_stride;
  *avgPool2d_node->window() = node_window;

  INFO(l) << "Canonicalize TFAvgPool pad = T " << avgPool2d_node->pad()->top() << ", L "
          << avgPool2d_node->pad()->left() << ", B " << avgPool2d_node->pad()->bottom() << ", R "
          << avgPool2d_node->pad()->right() << std::endl;

  // update graph
  auto node_A = node->value();

  // update connections
  feature_enc->input(node_A);
  avgPool2d_node->ifm(feature_enc);
  feature_dec->input(avgPool2d_node);

  // replace node
  replace(node).with(feature_dec);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool AvgPoolCanonicalizer::transform(TFAvgPool *node) const
{
  return canonicalize_avgpool2d(node->graph(), node);
}

} // namespace tf
} // namespace moco
