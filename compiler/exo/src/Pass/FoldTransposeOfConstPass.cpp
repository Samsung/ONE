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

#include "FoldTransposeOfConstPass.h"

#include "Check.h"

#include "Dialect/IR/TFLNodes.h"
#include "Dialect/IR/TFLNodeVisitor.h"

// TODO remove dependency to angkor
#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <oops/InternalExn.h>

namespace
{

/**
 * @brief   Check if node is TFLTranspose and its input is TFLConst
 * @return  Casted TFLTranspose for foldable candidate, nullptr otherwise
 */
locoex::TFLTranspose *as_candidate(loco::Node *node)
{
  auto transpose = dynamic_cast<locoex::TFLTranspose *>(node);
  if (not transpose)
    return nullptr;

  // Only accept Constant input of Transpose
  if (not dynamic_cast<locoex::TFLConst *>(transpose->a()))
    return nullptr;

  // Only accept Constant permutation of Transpose
  if (not dynamic_cast<locoex::TFLConst *>(transpose->perm()))
    return nullptr;

  return transpose;
}

nncc::core::ADT::tensor::Shape angkor_shape(locoex::TFLConst *node)
{
  nncc::core::ADT::tensor::Shape ret;

  ret.resize(node->rank());
  for (uint32_t axis = 0; axis < node->rank(); ++axis)
  {
    ret.dim(axis) = node->dim(axis).value();
  }

  return ret;
}

void fold_transpose_of_const(locoex::TFLTranspose *transpose)
{
  const loco::DataType FLOAT32 = loco::DataType::FLOAT32;
  const loco::DataType S32 = loco::DataType::S32;

  auto const_orig = dynamic_cast<locoex::TFLConst *>(transpose->a());
  auto perm = dynamic_cast<locoex::TFLConst *>(transpose->perm());

  // Exceptions
  {
    EXO_ASSERT(const_orig, "Only support for Transpose-Const pair");
    // TODO support other data types
    if (const_orig->dtype() != FLOAT32)
      INTERNAL_EXN_V("NYI for this data type", oops::to_uint32(const_orig->dtype()));

    EXO_ASSERT(perm, "Only support for constant permutation for Transpose");
    // TODO support other data types
    if (perm->dtype() != S32)
      INTERNAL_EXN_V("NYI for this data type", oops::to_uint32(perm->dtype()));

    auto okay = [&]() {
      if (perm->rank() != 1)
        return false;
      if (perm->dim(0).value() != const_orig->rank())
        return false;
      return true;
    };
    if (not okay())
      INTERNAL_EXN("Input and permutation for Transpose is not congruent");
  }

  uint32_t rank = const_orig->rank();

  // TFLConst to replace
  auto const_new = transpose->graph()->nodes()->create<locoex::TFLConst>();

  const_new->dtype(FLOAT32);
  const_new->rank(rank);
  const_new->size<FLOAT32>(const_orig->size<FLOAT32>());
  for (uint32_t axis = 0; axis < rank; ++axis)
    const_new->dim(axis) = const_orig->dim(perm->at<S32>(axis)).value();

  // TODO remove dependency to angkor
  auto shape_orig = angkor_shape(const_orig);
  auto shape_new = angkor_shape(const_new);

  nncc::core::ADT::tensor::LexicalLayout l;
  nncc::core::ADT::tensor::IndexEnumerator e{shape_new};

  for (; e.valid(); e.advance())
  {
    loco::TensorIndex index_new = e.current();
    loco::TensorIndex index_orig;

    // Set original index from matching new index
    index_orig.resize(rank);
    for (uint32_t axis = 0; axis < rank; ++axis)
      index_orig.at(perm->at<S32>(axis)) = index_new.at(axis);

    const_new->at<FLOAT32>(l.offset(shape_new, index_new)) =
      const_orig->at<FLOAT32>(l.offset(shape_orig, index_orig));
  }

  // replace
  loco::replace(transpose).with(const_new);
}

} // namespace

namespace exo
{

bool FoldTransposeOfConstPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto transpose = as_candidate(node))
    {
      fold_transpose_of_const(transpose);
      changed = true;
    }
  }

  return changed;
}

} // namespace exo
