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

#include "Flatten.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ops/ReshapeOp.h"

namespace mir_onnx
{

void convertFlattenV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  const auto axis = getAttributeValue<int64_t>(onnx_node, "axis", 1);
  assert(inputs.size() == 1);
  const auto &in_shape = inputs[0]->getShape();
  assert(axis <= in_shape.rank()); // A tensor of rank >= axis
  int32_t first_dim = 1, second_dim = 1;
  int32_t dim = 0;

  for (; dim < axis; dim++)
    first_dim *= in_shape.dim(dim);

  for (; dim < in_shape.rank(); dim++)
    second_dim *= in_shape.dim(dim);

  mir::Shape out_shape({first_dim, second_dim}); // Output 2D tensor

  auto result = createOp<mir::ops::ReshapeOp>(graph, inputs[0], out_shape)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertFlattenV9(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // Other type constraints
  convertFlattenV1(onnx_node, context);
}

} // namespace mir_onnx
