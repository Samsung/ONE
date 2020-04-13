/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Expand.h"

#include "ONNXHelpers.h"

#include "mir/ops/BroadcastOp.h"

namespace mir_onnx
{

void convertExpandV8(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  if (inputs[1]->getNode()->getType() != mir::Operation::Type::constant)
  {
    throw std::runtime_error{"Expand with non-constant input shape is not supported"};
  }

  auto target_shape = constantToShape(static_cast<mir::ops::ConstantOp *>(inputs[1]->getNode()));

  auto *result = createOp<mir::ops::BroadcastOp>(graph, inputs[0], target_shape)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
