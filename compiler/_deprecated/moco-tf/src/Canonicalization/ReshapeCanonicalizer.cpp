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

#include "ReshapeCanonicalizer.h"

#include <moco/IR/TFDialect.h>

#include <moco/Log.h>
#include <plier/tf/Convert.h>
#include <oops/UserExn.h>

#include <cassert>

namespace
{
using plier::tf::DataLayout;

/**
 * @brief  Check whether given 'new shape' arg is a fixed shape input for Reshape
 *
 * ConstNode can be moco::TFConst or loco::ConstGen
 */
template <typename ConstNode> bool is_fixed_shape_input(ConstNode *const_shape_input)
{
  if (const_shape_input == nullptr)
    return false;

  // Shape input should be integer tensor of rank 1, e.g. [2, 3, 4] or [3, -1]
  // TODO Support other possible data types, e.g. S64
  assert(const_shape_input->dtype() == loco::DataType::S32);
  assert(const_shape_input->rank() == 1);

  auto shape_rank = const_shape_input->dim(0).value();
  assert(shape_rank > 0);

  for (uint32_t axis = 0; axis < shape_rank; ++axis)
  {
    auto shape_dim = const_shape_input->template at<loco::DataType::S32>(axis);
    if (shape_dim == -1)
    {
      // has wildcard dimension, i.e. dynamic reshape
      return false;
    }
    if (!(shape_dim >= 1))
    {
      throw oops::UserExn("New shape of Reshape has invalid dimension");
    }
  }
  return true;
}

/// @note  Currently only supports to canonicalize Fixed Reshape
bool canonicalize_reshape(loco::Graph *graph, moco::TFReshape *node)
{
  LOGGER(l);
  INFO(l) << "TFNodeCanonicalize TFReshape begin";

  /**
   * This rule canonicalizes TFReshape only when its output shape is known at
   * compile time, i.e. fixed reshape case.
   * TODO Support other cases like dynamic reshape
   *
   * This will replace TFReshape + TFConst or Canonical ConstGen(as shape input)
   * node pair into Canonical Reshape<ReshapeType::Fixed>, or 'FixedReshape'.
   * Shape input (TFConst or Canonical ConstGen) should not have wildcard
   * dimension to be converted to FixedReshape.
   *
   * Before
   *           TFConst   (shape)
   *              or   ---
   *           ConstGen   \
   *                       \
   *           In --------- TFReshape ------- Out(s)
   *              (tensor)
   *
   * After
   *           TFConst
   *              or   ---
   *           ConstGen   \
   *                       \
   *             ---------- TFReshape
   *            /
   *           In -------- FixedReshape ----- Out(s)
   */

  // create loco node to replace
  auto fixed_reshape = graph->nodes()->create<loco::FixedReshape>();

  // Supports 2 cases for Reshape's shape input:
  //     TF-dialect TFConst or Canonical ConstGen
  loco::Node *shape_input = node->shape();
  auto tfconst_shape_input = dynamic_cast<moco::TFConst *>(shape_input);
  auto constgen_shape_input = dynamic_cast<loco::ConstGen *>(shape_input);

  if (tfconst_shape_input)
  {
    // Only support fixed reshape
    // TODO support dynamic reshape
    if (!(is_fixed_shape_input(tfconst_shape_input)))
    {
      throw oops::UserExn("Supports only fixed reshape", node->name());
    }

    auto rank = tfconst_shape_input->dim(0).value();
    fixed_reshape->rank(rank);
    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      fixed_reshape->dim(axis) = tfconst_shape_input->at<loco::DataType::S32>(axis);
    }
  }
  else if (constgen_shape_input)
  {
    // ditto
    if (!(is_fixed_shape_input(constgen_shape_input)))
    {
      throw oops::UserExn("Supports only fixed reshape", node->name());
    }

    auto rank = constgen_shape_input->dim(0).value();
    fixed_reshape->rank(rank);
    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      fixed_reshape->dim(axis) = constgen_shape_input->at<loco::DataType::S32>(axis);
    }
  }
  else
  {
    // TODO support dynamic reshape from not const node
    throw oops::UserExn("Supports only const node as input shape", node->name());
  }

  // replace
  auto in = node->tensor();
  fixed_reshape->input(in);

  replace(node).with(fixed_reshape);

  INFO(l) << "TFNodeCanonicalize TFReshape done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool ReshapeCanonicalizer::transform(TFReshape *node) const
{
  return canonicalize_reshape(node->graph(), node);
}

} // namespace tf
} // namespace moco
