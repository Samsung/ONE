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

#include "Relu.h"

#include "ONNXHelpers.h"

#include "mir/ops/ReluOp.h"

namespace mir_onnx
{

static void convertRelu(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();
  assert(inputs.size() == 1);
  auto result = createOp<mir::ops::ReluOp>(graph, inputs[0])->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertReluV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  convertRelu(onnx_node, context);
}

void convertReluV6(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  convertRelu(onnx_node, context);
}

} // namespace mir_onnx
