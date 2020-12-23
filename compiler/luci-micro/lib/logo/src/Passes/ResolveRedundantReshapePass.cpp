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

#include <logo/ResolveRedundantReshapePass.h>

#include <loco/Service/ShapeInference.h>

#include <loco.h>

#include <cassert>

namespace
{

bool shape_inference_done(loco::FixedReshape *reshape)
{
  return loco::shape_known(reshape) && loco::shape_known(reshape->input());
}

bool are_same_tensor_shapes(const loco::NodeShape &lhs, const loco::NodeShape &rhs)
{
  assert(lhs.domain() == loco::Domain::Tensor);
  assert(rhs.domain() == loco::Domain::Tensor);

  auto lts = lhs.as<loco::TensorShape>();
  auto rts = rhs.as<loco::TensorShape>();

  if (lts.rank() != rts.rank())
    return false;

  for (uint32_t axis = 0; axis < lts.rank(); ++axis)
  {
    assert(lts.dim(axis).known());
    assert(rts.dim(axis).known());
    if (lts.dim(axis).value() != rts.dim(axis).value())
      return false;
  }
  return true;
}

/// @return  true when 'reshape' has same input and output shape
bool is_redundant_reshape(loco::FixedReshape *reshape)
{
  auto input_shape = loco::shape_get(reshape->input());
  auto output_shape = loco::shape_get(reshape);

  // Note that FixedReshape's input and output are always tensor
  return are_same_tensor_shapes(input_shape, output_shape);
}

} // namespace

namespace logo
{

/**
 * @brief  Bypass redundant FixedReshape
 *
 * Before:
 *
 *   In ----- FixedReshape ----- [Out]*
 *
 * After:
 *
 *   In ------------------------ [Out]*
 *    \
 *     ------ FixedReshape
 */
bool ResolveRedundantReshapePass::run(loco::Graph *graph)
{
  bool changed = false;
  for (auto node : loco::postorder_traversal(loco::output_nodes(graph)))
  {
    if (auto reshape = dynamic_cast<loco::FixedReshape *>(node))
    {
      if (shape_inference_done(reshape))
      {
        if (is_redundant_reshape(reshape))
        {
          replace(reshape).with(reshape->input());
          changed = true;
        }
      }
    }
  }

  return changed;
}

} // namespace logo
