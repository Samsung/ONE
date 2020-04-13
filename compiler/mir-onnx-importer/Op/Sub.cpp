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

#include "Sub.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ops/SubOp.h"

namespace mir_onnx
{

void convertSubV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // consumed_inputs attribute not used
  convertSubV6(onnx_node, context);
}

void convertSubV6(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // broadcast attribute not used
  const auto *axis = findAttribute(onnx_node, "axis");
  if (axis != nullptr)
    throw std::runtime_error("Not supported axis attribute in Sub operation!");

  convertSubV7(onnx_node, context);
}

void convertSubV7(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  auto result = createOp<mir::ops::SubOp>(graph, inputs[0], inputs[1])->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
