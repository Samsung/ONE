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

#include "BiasAddCanonicalizer.h"

#include <moco/IR/TFDialect.h>

#include <moco/Names.h>
#include <moco/Log.h>
#include <plier/tf/Convert.h>

namespace
{
using plier::tf::DataLayout;

bool canonicalize_biasadd(loco::Graph *graph, moco::TFBiasAdd *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFBiasAdd node with Canonical BiasEncode + TensorBiasAdd
   *
   *       Before
   *                 A -- TFBiasAdd - C
   *                 B -/
   *
   *       After
   *                 A -- TFBiasAdd -
   *                 B -/
   *                 A --------------- TensorBiasAdd - C
   *                 B - BiasEncode -/
   *
   *       Where
   *                 A : value of TFBiasAdd
   *                 B : bias of TFBiasAdd
   *                 C : a node that uses TFBiasAdd as an input
   *                 TFBiasAdd is disconnected from node C
   *                 A and B are drawn twice to simplify the diagram
   */

  INFO(l) << "TFNodeCanonicalize TFBiasAdd begin";

  // tensorflow data_format: one of NHWC or NCHW.
  auto data_layout = plier::tf::as_data_layout(node->data_layout());

  // creating loco nodes
  auto bias_enc = graph->nodes()->create<loco::BiasEncode>();

  auto bias_add = graph->nodes()->create<loco::TensorBiasAdd>();
  {
    if (data_layout == DataLayout::NHWC)
    {
      INFO(l) << "TFNodeCanonicalize TFBiasAdd axis 3";
      bias_add->axis(3);
    }
    else if (data_layout == DataLayout::NCHW)
    {
      INFO(l) << "TFNodeCanonicalize TFBiasAdd axis 1";
      bias_add->axis(1); // Channel
      // Note: the following descrition of TF 1.13 at
      // https://www.tensorflow.org/api_docs/python/tf/nn/bias_add seems wrong:
      // "bias: A 1-D Tensor with size matching the last dimension of value."
      // because providing the size of W (last dimension) to bias throws an error with TensorFlow
    }
  }

  auto node_A = node->value();
  auto node_B = node->bias();

  // update connections
  bias_add->value(node_A);
  bias_add->bias(bias_enc);
  bias_enc->input(node_B);

  // replace old with new : about C in above note
  replace(node).with(bias_add);

  INFO(l) << "TFNodeCanonicalize TFBiasAdd done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool BiasAddCanonicalizer::transform(TFBiasAdd *node) const
{
  return canonicalize_biasadd(node->graph(), node);
}

} // namespace tf
} // namespace moco
