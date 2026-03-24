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

#include "passes/optimizations/ConstantFoldTranspose.h"
#include "passes/optimizations/OptimizationUtils.h"
#include "mir/GraphPatternMatcher.h"
#include "mir/ShapeRange.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/TransposeOp.h"

#include <cstring>

using namespace nnc;
using namespace mir;

// Copy & paste from interpreter backend.
// TODO Extract this to a common place and use in both interpreter and optimizations.
static void transpose(const TensorVariant &input, TensorVariant &res,
                      const std::vector<std::size_t> &axis_order)
{
  const auto &input_shape = input.getShape();
  const int num_axes = static_cast<int>(axis_order.size());
  assert(num_axes == input_shape.rank());

  ShapeRange in_range(input_shape);
  Index out_index(input_shape.rank());

  const size_t elem_size = input.getElementSize();

  for (const auto &in_index : in_range)
  {
    for (int i = 0; i < num_axes; ++i)
      out_index.at(i) = in_index.at(axis_order[i]);

    std::memcpy(res.at(out_index), input.at(in_index), elem_size);
  }
}

PassData ConstantFoldTranspose::run(PassData data)
{
  auto graph = static_cast<Graph *>(data);

  GraphPatternMatcher matcher(graph);
  auto is_constant = [](const Operation *op) { return op->getType() == Operation::Type::constant; };
  auto is_transpose = [](const Operation *op) {
    return op->getType() == Operation::Type::transpose;
  };

  auto matches = matcher.matchEdge(is_constant, is_transpose);
  while (!matches.empty())
  {
    for (const auto &match : matches)
    {
      auto constant_op = dynamic_cast<ops::ConstantOp *>(match.first);
      auto transpose_op = dynamic_cast<ops::TransposeOp *>(match.second);

      const auto elem_type = constant_op->getValue().getElementType();
      const auto &out_shape = transpose_op->getOutputShape(0);
      TensorType res_type(elem_type, out_shape);
      if (constant_op->getOutput(0)->getType().isQuantized())
        res_type.setQuantization(constant_op->getOutput(0)->getType().getQuantization());

      TensorVariant res(res_type);
      transpose(constant_op->getValue(), res, transpose_op->getAxisOrder());

      auto new_op = graph->create<ops::ConstantOp>(res);

      graph->replaceNode(transpose_op, new_op);
      opt_util::removeNodeIfUnused(graph, constant_op);
    }
    matches = matcher.matchEdge(is_constant, is_transpose);
  }
  return graph;
}
