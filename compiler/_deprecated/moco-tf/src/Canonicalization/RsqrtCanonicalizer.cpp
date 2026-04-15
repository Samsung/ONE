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

#include "RsqrtCanonicalizer.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include <moco/Log.h>

#include <loco/Service/TypeInference.h>

#include <oops/UserExn.h>

namespace
{

template <typename T>
bool prepare_const_gen(loco::ConstGen *const_node, const loco::TensorShape &tensorshape, T value);

template <>
bool prepare_const_gen<float>(loco::ConstGen *const_node, const loco::TensorShape &tensorshape,
                              float value)
{
  LOGGER(l);

  uint32_t const_num_elements = 1;

  auto dtype = loco::DataType::FLOAT32;
  const_node->dtype(dtype);

  auto rank = tensorshape.rank();
  const_node->rank(rank);
  for (uint32_t r = 0; r < rank; ++r)
  {
    if (tensorshape.dim(r).known())
      const_node->dim(r) = tensorshape.dim(r);
    else
      return false;

    assert(tensorshape.dim(r).value() > 0);

    const_num_elements *= tensorshape.dim(r).value();
  }

  INFO(l) << "prepare_const_gen : Elements = " << const_num_elements;

  const_node->size<loco::DataType::FLOAT32>(const_num_elements);
  for (uint32_t i = 0; i < const_num_elements; ++i)
  {
    const_node->at<loco::DataType::FLOAT32>(i) = value;
  }

  return true;
}

bool canonicalize_rsqrt(loco::Graph *graph, moco::TFRsqrt *node)
{
  /**
   * @note This will replace TFRsqrt node with Canonical EltwiseSqrt + EltwiseRealDiv
   *
   *       Before
   *                 A --- TFRsqrt -- C
   *       After
   *                    +- TFRsqrt --
   *                    |
   *                    |   ConstGen --+
   *                    |               \
   *                 A -+- EltwiseSqrt -- EltwiseDiv -- C
   *
   *       Where
   *                 A : features of TFRsqrt
   *                 C : a node that uses TFSqrt as an input
   *                 TFRsqrt is disconnected from C
   *                 TFRsqrt is converted to 1 / EltwiseSqrt
   */

  auto nodeshape = moco::node_shape(node);
  if (nodeshape.domain() == loco::Domain::Unknown)
  {
    // We need this shape information
    assert(false); // this shouldn't happen, let's add an alarm
    return false;
  }
  auto tensorshape = nodeshape.as<loco::TensorShape>();

  if (!loco::dtype_known(node))
  {
    // We need type of this node
    return false;
  }

  auto sqrt_node = graph->nodes()->create<loco::EltwiseSqrt>();
  auto eltdiv_node = graph->nodes()->create<loco::EltwiseDiv>();
  auto const_node = graph->nodes()->create<loco::ConstGen>();

  auto dtype = loco::dtype_get(node);

  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      if (!prepare_const_gen<float>(const_node, tensorshape, 1.0f))
        throw oops::UserExn("Cannot handle unknown shape", node->name());
      break;

    default:
      throw oops::UserExn("Unsupported data type", node->name());
  }

  auto node_A = node->x();

  // update connections
  sqrt_node->input(node_A);
  eltdiv_node->lhs(const_node);
  eltdiv_node->rhs(sqrt_node);

  // replace node
  replace(node).with(eltdiv_node);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool RsqrtCanonicalizer::transform(TFRsqrt *node) const
{
  return canonicalize_rsqrt(node->graph(), node);
}

} // namespace tf
} // namespace moco
