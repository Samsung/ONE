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

#include "SqueezeCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include <moco/Log.h>

namespace
{

bool canonicalize_squeeze_to_reshape(loco::Graph *graph, moco::TFSqueeze *node)
{
  LOGGER(l);

  INFO(l) << "TFNodeCanonicalize TFSqueeze begin";

  /**
   * This will replace shape inferred TFSqueeze node into canonical FixedReshape
   *
   * Before
   *           In ---- TFSqueeze ---- Out(s)
   *
   * After
   *             ------ TFSqueeze
   *            /
   *           In ---- FixedReshape ----- Out(s)
   */

  auto nodeshape = moco::node_shape(node);
  // canonicalize into FixedReshape is valid when squeeze has shape info
  // TODO Support general Squeeze case
  assert(nodeshape.domain() != loco::Domain::Unknown);

  auto squeeze_tensor_shape = nodeshape.as<loco::TensorShape>();

  // Create loco node to replace
  auto reshape = graph->nodes()->create<loco::FixedReshape>();

  // Copy shape
  reshape->rank(squeeze_tensor_shape.rank());
  for (uint32_t axis = 0; axis < squeeze_tensor_shape.rank(); ++axis)
  {
    assert(squeeze_tensor_shape.dim(axis).known());
    reshape->dim(axis) = squeeze_tensor_shape.dim(axis);
  }

  // replace
  auto in = node->input();
  reshape->input(in);
  replace(node).with(reshape);

  INFO(l) << "TFNodeCanonicalize TFSqueeze done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool SqueezeCanonicalizer::transform(TFSqueeze *node) const
{
  return canonicalize_squeeze_to_reshape(node->graph(), node);
}

} // namespace tf
} // namespace moco
