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

#include "SoftmaxCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include <moco/Log.h>

namespace
{

bool canonicalize_softmax(loco::Graph *graph, moco::TFSoftmax *node)
{
  LOGGER(l);

  INFO(l) << "TFNodeCanonicalize TFSoftmax begin";

  /**
   * This will replace shape inferred TFSoftmax node into canonical TensorSoftmax
   *
   * Before
   *           In ---- TFSoftmax ---- Out(s)
   *
   * After
   *             ------ TFSoftmax
   *            /
   *           In ---- TensorSoftmax ----- Out(s)
   */

  auto nodeshape = moco::node_shape(node);
  // Canonicalization into TensorSoftmax is valid when softmax has shape info
  assert(nodeshape.domain() != loco::Domain::Unknown);

  auto softmax_tensor_shape = nodeshape.as<loco::TensorShape>();

  // Create loco node to replace
  auto softmax = graph->nodes()->create<loco::TensorSoftmax>();

  // replace
  auto in = node->logits();
  softmax->input(in);
  softmax->axis(softmax_tensor_shape.rank() - 1);
  replace(node).with(softmax);

  INFO(l) << "TFNodeCanonicalize TFSoftmax done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool SoftmaxCanonicalizer::transform(TFSoftmax *node) const
{
  return canonicalize_softmax(node->graph(), node);
}

} // namespace tf
} // namespace moco
