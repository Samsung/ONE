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

#include "MatMul.h"

#include "ONNXHelpers.h"

#include "mir/ops/FullyConnectedOp.h"

namespace mir_onnx
{

void convertMatMulV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 2);
  auto A = inputs[0];
  auto B = inputs[1];
  // MatMul multiply N-dimentional matrix
  // FullyConnected layer multiply only 2-dimentional matrix
  if (A->getShape().rank() != 2 || B->getShape().rank() != 2)
    throw std::runtime_error("Supported only 2D matrix multiplying!");
  // Calculate A * B.
  auto result = createOp<mir::ops::FullyConnectedOp>(graph, A, B)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertMatMulV9(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // Other type constraints
  convertMatMulV1(onnx_node, context);
}

} // namespace mir_onnx
