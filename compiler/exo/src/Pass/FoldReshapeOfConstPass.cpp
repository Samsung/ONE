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

#include "FoldReshapeOfConstPass.h"

#include "Check.h"

#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLNodeVisitor.h"

#include <loco/Service/ShapeInference.h>

#include <oops/InternalExn.h>

namespace
{

/**
 * @brief   Check if node is TFLReshape and its input is TFLConst
 * @return  Casted TFLReshape for foldable candidate, nullptr otherwise
 */
locoex::TFLReshape *as_candidate(loco::Node *node)
{
  auto reshape = dynamic_cast<locoex::TFLReshape *>(node);
  if (not reshape)
    return nullptr;

  // Only accept Constant input of Reshape
  if (not dynamic_cast<locoex::TFLConst *>(reshape->tensor()))
    return nullptr;

  return reshape;
}

uint32_t volume(loco::Node *tensor_node)
{
  auto shape = loco::shape_get(tensor_node).as<loco::TensorShape>();

  uint32_t vol = 1;
  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
    vol *= shape.dim(axis).value();

  return vol;
}

void fold_reshape_of_const(locoex::TFLReshape *reshape)
{
  const loco::DataType FLOAT32 = loco::DataType::FLOAT32;

  auto const_orig = dynamic_cast<locoex::TFLConst *>(reshape->tensor());

  // Exceptions
  {
    EXO_ASSERT(const_orig, "Only support for Reshape-Const pair");
    // TODO support other data types
    if (const_orig->dtype() != FLOAT32)
      INTERNAL_EXN_V("NYI for this data type", oops::to_uint32(const_orig->dtype()));

    if (volume(const_orig) != volume(reshape))
      INTERNAL_EXN("New shape of Reshape is not matched");
  }

  auto new_shape = loco::shape_get(reshape).as<loco::TensorShape>();

  // TFLConst to replace
  auto const_new = reshape->graph()->nodes()->create<locoex::TFLConst>();

  const_new->dtype(FLOAT32);
  const_new->rank(new_shape.rank());
  const_new->size<FLOAT32>(const_orig->size<FLOAT32>());
  for (uint32_t axis = 0; axis < new_shape.rank(); ++axis)
    const_new->dim(axis) = new_shape.dim(axis);

  for (uint32_t i = 0; i < const_new->size<FLOAT32>(); ++i)
  {
    const_new->at<FLOAT32>(i) = const_orig->at<FLOAT32>(i);
  }

  // replace
  loco::replace(reshape).with(const_new);
}

} // namespace

namespace exo
{

bool FoldReshapeOfConstPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto reshape = as_candidate(node))
    {
      fold_reshape_of_const(reshape);
      changed = true;
    }
  }

  return changed;
}

} // namespace exo
