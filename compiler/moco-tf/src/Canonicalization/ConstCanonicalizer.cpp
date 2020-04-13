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

#include "ConstCanonicalizer.h"

#include <moco/IR/TFDialect.h>

#include <moco/Names.h>
#include <moco/Log.h>

#include <oops/UserExn.h>

namespace
{

bool canonicalize_const(loco::Graph *graph, moco::TFConst *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFConst node with Canonical Const
   *
   *       Before
   *                 TFConst -- C
   *
   *       After
   *                 TFConst -
   *                 ConstGen -- C
   *
   *       Where
   *                 C : a node that uses TFConst as an input
   *                 TFConst is disconnected from other nodes
   */

  INFO(l) << "TFNodeCanonicalize TFConst begin";

  auto const_node = graph->nodes()->create<loco::ConstGen>();

  // copy properties
  auto dtype = node->dtype();
  const_node->dtype(dtype);

  auto rank = node->rank();

  if (rank == 0)
  {
    // This routine implements a workaround that converts a scalar constant (rank-0 tensor)
    // into a rank-1 tensor of shape [1].
    //
    // TODO Revise this implementation later
    const_node->rank(1);
    const_node->dim(0) = 1;
  }
  else
  {
    const_node->rank(rank);

    for (uint32_t r = 0; r < rank; ++r)
    {
      if (node->dim(r).known())
        const_node->dim(r) = node->dim(r);
      else
        const_node->dim(r).unset();
    }
  }

  switch (dtype)
  {
    case loco::DataType::S32:
    {
      uint32_t input_elements = node->size<loco::DataType::S32>();
      const_node->size<loco::DataType::S32>(input_elements);
      for (uint32_t i = 0; i < input_elements; ++i)
      {
        const_node->at<loco::DataType::S32>(i) = node->at<loco::DataType::S32>(i);
      }
      break;
    }
    case loco::DataType::FLOAT32:
    {
      uint32_t input_elements = node->size<loco::DataType::FLOAT32>();
      const_node->size<loco::DataType::FLOAT32>(input_elements);
      for (uint32_t i = 0; i < input_elements; ++i)
      {
        const_node->at<loco::DataType::FLOAT32>(i) = node->at<loco::DataType::FLOAT32>(i);
      }
      break;
    }
    default:
      throw oops::UserExn("Const has unsupported data type", node->name());
  }

  // update graph
  replace(node).with(const_node);

  INFO(l) << "TFNodeCanonicalize TFConst done";

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool ConstCanonicalizer::transform(TFConst *node) const
{
  return canonicalize_const(node->graph(), node);
}

} // namespace tf
} // namespace moco
