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

#include "moco/Pass/Passes/ConstantFoldAdd.h"

#include "ConstantFoldHelper.h"

#include <moco/IR/Nodes/TFAdd.h>
#include <moco/IR/Nodes/TFConst.h>

#include <moco/Support/NodeAs.h>

namespace
{

struct Func final : public moco::BinaryFunc
{
  float apply(float lhs, float rhs) const { return lhs + rhs; }
  int32_t apply(int32_t lhs, int32_t rhs) const { return lhs + rhs; }
};

bool constantfold_add(moco::TFAdd *node)
{
  auto x_const = moco::as<moco::TFConst>(node->x());
  auto y_const = moco::as<moco::TFConst>(node->y());
  if (x_const == nullptr || y_const == nullptr)
    return false;

  if (x_const->dtype() != y_const->dtype())
    return false;
  // TODO support other types
  if (x_const->dtype() != loco::DataType::S32 && x_const->dtype() != loco::DataType::FLOAT32)
    return false;

  // NOTE we support limited shape of elementwise add or add with a scalar.
  //      valid_shape_for_constfold_binary_op() explains limited shape.
  auto x_shape = moco::tensor_shape(x_const);
  auto y_shape = moco::tensor_shape(y_const);
  if (!moco::valid_shape_for_constfold_binary_op(x_shape, y_shape))
    return false;

  loco::TensorShape output_shape;
  if (y_shape.rank() == 0 || y_shape.rank() == 1)
    output_shape = x_shape;
  else
    output_shape = y_shape;

  auto graph = node->graph();
  auto output_const = moco::new_const(graph, output_shape, x_const->dtype());
  Func f;

  if (x_const->dtype() == loco::DataType::S32)
  {
    moco::apply_binary<int32_t>(x_const, y_const, output_const, f);
  }
  else if (x_const->dtype() == loco::DataType::FLOAT32)
  {
    moco::apply_binary<float>(x_const, y_const, output_const, f);
  }

  // replace
  loco::replace(node).with(output_const);

  return true;
}

} // namespace

namespace moco
{

/**
 * @note This will Replace TFAdd with TFConst when inputs are TFConst
 *
 *       Before
 *                 A --- TFAdd --- C
 *                 B --/
 *       After
 *                 A --- TFAdd
 *                 B --/
 *                       TFConst ---------- C
 *       Where
 *                 A,B : inputs of TFAdd
 *                 C : a node that uses TFAdd as an input
 *                 TFAdd is disconnected from C
 *                 Nodes are drawn multiple times to simplify the diagram
 */
bool ConstantFoldAdd::run(loco::Graph *graph)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(graph)))
  {
    if (auto add_node = as<moco::TFAdd>(node))
    {
      if (constantfold_add(add_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace moco
