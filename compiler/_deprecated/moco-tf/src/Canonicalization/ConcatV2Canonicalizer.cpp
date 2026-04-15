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

#include "ConcatV2Canonicalizer.h"
#include "LogHelper.h"

#include <moco/IR/TFDialect.h>
#include <moco/Support/TFShapeInferenceHelper.h>

#include <moco/Log.h>

#include <loco/Service/ShapeInference.h>

#include <oops/UserExn.h>

namespace
{

using namespace moco::tf;

bool scalar_value(moco::TFConst *node, int32_t &ret)
{
  auto nodeshape = node_shape(node);
  if (!(node->dtype() == loco::DataType::S32))
    return false;

  auto tensor_shape = nodeshape.as<loco::TensorShape>();
  if (!(tensor_shape.rank() == 0 || tensor_shape.rank() == 1))
    return false;

  ret = node->at<loco::DataType::S32>(0);

  return true;
}

bool canonicalize_concat(loco::Graph *graph, moco::TFConcatV2 *node)
{
  LOGGER(l);

  /**
   * @note This will replace TFConcatV2 node with (series of) Canonical
   *       TensorConcat. Below diagram is an example of three inputs
   *
   *       Before
   *                 A --- TFConcatV2 -- C
   *                 B --/
   *                 N --/
   *                 X --/
   *       After
   *                 A --- TFConcatV2
   *                 B --/
   *                 N --/
   *                 X --/
   *                 A --- TensorConcat -- TensorConcat -- C
   *                 B --/               /
   *                 N -----------------/
   *
   *       Where
   *                 A : first value of TFConcatV2
   *                 B : second value of TFConcatV2
   *                 N : third or N'th value of TFConcatV2
   *                 X : axis node of TFConcatV2
   *                 C : a node that uses TFConcatV2 as an input
   *                 TFConcatV2 is disconnected from C
   *                 To simplify the diagram in 'After', A, B, N are drawn
   *                 multiple times but they are same nodes.
   */

  const int num_values = node->num_values();
  assert(num_values >= 2);

  // get axis absolute value
  auto value_a = node->values(0);
  if (!loco::shape_known(value_a))
    return false;

  uint32_t node_rank = 0;
  {
    auto value_a_shape = moco::node_shape(value_a);
    assert(value_a_shape.domain() == loco::Domain::Tensor);

    auto value_a_tensor_shape = value_a_shape.as<loco::TensorShape>();
    node_rank = value_a_tensor_shape.rank();
  }

  int32_t axis_value = 0;
  {
    // axis should be TFConst
    auto axis_node = node->axis();
    auto tfconst = dynamic_cast<moco::TFConst *>(axis_node);
    if (tfconst == nullptr)
    {
      // TODO Check this: this error can be from TFOptimizatier.
      throw oops::UserExn("ConcatV2 node has invalid input for axis", node->name());
    }
    auto result = scalar_value(tfconst, axis_value);
    if (!result)
    {
      // TODO Check this: this error can be from TFOptimizatier.
      throw oops::UserExn("ConcatV2 node has invalid input for axis", node->name());
    }
  }
  uint32_t axis_absolute = (axis_value >= 0) ? axis_value : (int32_t)node_rank + axis_value;

  INFO(l) << "canonicalize_concat axis(" << axis_absolute << "), value(" << axis_value << "), rank("
          << node_rank << ")";

  // Convert series of TensorConcat if num_values > 2
  auto concat_node = graph->nodes()->create<loco::TensorConcat>();
  concat_node->lhs(node->values(0));
  concat_node->rhs(node->values(1));
  concat_node->axis(axis_absolute);

  loco::TensorConcat *last_concat = concat_node;
  for (int ni = 2; ni < num_values; ++ni)
  {
    auto concat_node_next = graph->nodes()->create<loco::TensorConcat>();

    concat_node_next->lhs(last_concat);
    concat_node_next->rhs(node->values(ni));
    concat_node_next->axis(axis_absolute);

    // update last concat node
    last_concat = concat_node_next;
  }

  // replace node
  replace(node).with(last_concat);

  return true;
}

} // namespace

namespace moco
{
namespace tf
{

bool ConcatV2Canonicalizer::transform(TFConcatV2 *node) const
{
  return canonicalize_concat(node->graph(), node);
}

} // namespace tf
} // namespace moco
