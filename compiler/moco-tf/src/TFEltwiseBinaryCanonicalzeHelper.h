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

#ifndef __TF_ELTWISE_BINARY_CANONICALIZE_HELPER_H__
#define __TF_ELTWISE_BINARY_CANONICALIZE_HELPER_H__

#include <moco/IR/TFDialect.h>
#include <moco/IR/TFNodes.h>

#include "CanonicalEltwiseInputConnector.h"
#include "BroadcastHelper.h"

#include <loco/IR/Nodes.h>
#include <loco/IR/NodeShape.h>
#include <loco/Service/ShapeInference.h>

#include <fipe.h>

namespace
{

template <typename TFNodeT> struct EltwiseBinaryCanonicalizationRule;

template <> struct EltwiseBinaryCanonicalizationRule<moco::TFAdd>
{
  using CanonicalNode = loco::EltwiseAdd;
};

template <> struct EltwiseBinaryCanonicalizationRule<moco::TFSub>
{
  using CanonicalNode = loco::EltwiseSub;
};

template <> struct EltwiseBinaryCanonicalizationRule<moco::TFMaximum>
{
  using CanonicalNode = loco::EltwiseMax;
};

template <> struct EltwiseBinaryCanonicalizationRule<moco::TFMul>
{
  using CanonicalNode = loco::EltwiseMul;
};

template <> struct EltwiseBinaryCanonicalizationRule<moco::TFRealDiv>
{
  using CanonicalNode = loco::EltwiseDiv;
};

template <typename TFNode> bool canonicalize_eltwise_binary_node(TFNode *node)
{
  auto graph = node->graph();

  /**
   * This will replace T/F Eltwise Binary node with a corresponding Canonical Eltwise node
   *
   * BEFORE
   *   A --- T/F Node --- C
   *         /
   *   B ----
   *
   * AFTER
   *   A --- T/F Node ---
   *         /
   *   B ----
   *
   *   A --- [FixedReshape] --- [TensorBroadcast] --- Canonical Node -- C
   *                                                  /
   *   B --- [FixedReshape] --- [TensorBroadcast] ----
   *
   * NOTE
   *   - [...] means optional node. They may or may not be created during this procedure.
   *   - T/F Node is disconnected from C after transformation.
   */

  using CanonicalNodeT = typename EltwiseBinaryCanonicalizationRule<TFNode>::CanonicalNode;

  auto node_A = node->x();
  auto node_B = node->y();

  if (!loco::shape_known(node_A) || !loco::shape_known(node_B))
    return false;
  if (!loco::shape_known(node))
    return false;

  auto out_shape = loco::shape_get(node).template as<loco::TensorShape>();

  // Create a node
  auto canonical_node = graph->nodes()->template create<CanonicalNodeT>();

  using moco::tf::broadcast_to;
  using moco::tf::eltwise::binary::connect_to;

  // update connections
  std::make_pair(node_A, node_B) | broadcast_to(out_shape) | connect_to(canonical_node);

  // replace node
  replace(node).with(canonical_node);

  return true;
}

} // namespace

#endif // __TF_ELTWISE_BINARY_CANONICALIZE_HELPER_H__
