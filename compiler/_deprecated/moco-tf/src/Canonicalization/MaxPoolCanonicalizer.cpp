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

#include "MaxPoolCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include "CodecHelper.h"

#include <moco/Log.h>

namespace
{

bool canonicalize_maxpool2d(loco::Graph *graph, moco::TFMaxPool *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFMaxPool node with Canonical FeatureEncode +
   *       MaxPool2D + FeatureDecode
   *
   *       Before
   *                 A -- TFMaxPool -- C
   *
   *       After
   *                    +- TFMaxPool --
   *                    |
   *                 A -+- FeatureEncode -- MaxPool2D -- FeatureDecode -- C
   *
   *       Where
   *                 A : value of TFMaxPool
   *                 C : a node that uses TFMaxPool as an input
   *                 TFMaxPool is disconnected from other nodes
   */

  auto data_layout = plier::tf::as_data_layout(node->data_layout());

  auto feature_enc = graph->nodes()->create<loco::FeatureEncode>();
  auto maxPool2d_node = graph->nodes()->create<loco::MaxPool2D>();
  auto feature_dec = graph->nodes()->create<loco::FeatureDecode>();

  set_feature_enc(feature_enc, data_layout);
  set_feature_dec(feature_dec, data_layout);

  // paddata to pad
  auto input_shape = moco::node_shape(node->input());
  assert(input_shape.domain() != loco::Domain::Unknown);

  auto node_stride = moco::stride_of(node->strides(), node->data_layout());
  auto node_window = moco::window_of(node->ksize(), node->data_layout());

  moco::Padding2DInference infer_padding2d;

  infer_padding2d.padding(node->padding());
  infer_padding2d.stride(node_stride);
  infer_padding2d.window(node_window);

  auto input_feature_shape = moco::as_feature_shape(input_shape, node->data_layout());
  auto input_plane_shape = moco::make_plane_shape(input_feature_shape);

  *maxPool2d_node->pad() = infer_padding2d(input_plane_shape);
  *maxPool2d_node->stride() = node_stride;
  *maxPool2d_node->window() = node_window;

  INFO(l) << "Canonicalize TFMaxPool pad = T " << maxPool2d_node->pad()->top() << ", L "
          << maxPool2d_node->pad()->left() << ", B " << maxPool2d_node->pad()->bottom() << ", R "
          << maxPool2d_node->pad()->right() << std::endl;

  // update graph
  auto node_A = node->input();

  // update connections
  feature_enc->input(node_A);
  maxPool2d_node->ifm(feature_enc);
  feature_dec->input(maxPool2d_node);

  // replace node
  replace(node).with(feature_dec);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool MaxPoolCanonicalizer::transform(TFMaxPool *node) const
{
  return canonicalize_maxpool2d(node->graph(), node);
}

} // namespace tf
} // namespace moco
