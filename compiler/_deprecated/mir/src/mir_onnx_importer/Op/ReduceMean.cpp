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

#include "ReduceMean.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ops/ReduceMeanOp.h"

#include <numeric>

namespace mir_onnx
{

void convertReduceMeanV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  const auto inputs = context->getNodeInputs(onnx_node);
  assert(inputs.size() == 1);

  const auto axes = getAttributeValue<std::vector<std::int64_t>>(onnx_node, "axes");
  const auto keepdims = getAttributeValue<int64_t>(onnx_node, "keepdims", 1);

  std::vector<int32_t> reduce_dims;
  if (axes.empty())
  { // reduce over all dimensions
    reduce_dims.resize(inputs[0]->getShape().rank());
    std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  }
  else
  {
    auto rank = inputs[0]->getShape().rank();

    std::transform(axes.begin(), axes.end(), std::back_inserter(reduce_dims),
                   [rank](int64_t axis) { return axis < 0 ? axis + rank : axis; });
  }
  // Keep the reduced dimension or not, default 1 mean keep reduced dimension.
  bool keep_dims = static_cast<bool>(keepdims);

  mir::Graph *graph = context->getGraph();
  auto result =
    createOp<mir::ops::ReduceMeanOp>(graph, inputs[0], reduce_dims, keep_dims)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
