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

#include "Transpose.h"
#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ops/TransposeOp.h"

#include <numeric>

namespace mir_onnx
{

void convertTransposeV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  const auto inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 1);
  auto input = inputs[0];

  const int num_axes = input->getShape().rank();
  std::vector<std::size_t> axis_order(num_axes);
  const auto *perm_attr = findAttribute(onnx_node, "perm");

  if (perm_attr == nullptr)
  {
    // Reverse the dimensions.
    std::iota(axis_order.rbegin(), axis_order.rend(), 0);
  }
  else
  {
    const auto perm = getAttributeValue<std::vector<std::int64_t>>(*perm_attr);
    assert(static_cast<int>(perm.size()) == num_axes);
    std::copy(perm.cbegin(), perm.cend(), axis_order.begin());
  }

  auto result = createOp<mir::ops::TransposeOp>(graph, input, axis_order)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
