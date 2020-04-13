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

#include "Concat.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ops/ConcatOp.h"

namespace mir_onnx
{

void convertConcatV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  const auto axis = getAttributeValue<int64_t>(onnx_node, "axis", 1);

  auto result = createOp<mir::ops::ConcatOp>(graph, inputs, axis)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertConcatV4(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();
  // From version 4 axis attribute is required
  auto attr = findAttribute(onnx_node, "axis");
  if (!attr)
    throw std::runtime_error("Attribute axis is required!");
  int32_t axis = attr->i();

  auto result = createOp<mir::ops::ConcatOp>(graph, inputs, axis)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
