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

#include "Reciprocal.h"

#include "ONNXHelpers.h"

#include "mir/ops/ConstantOp.h"
#include "mir/ops/DivOp.h"

namespace mir_onnx
{

static void convertReciprocal(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 1);
  auto input = inputs[0];

  const float one_value = 1.0f;
  mir::TensorVariant one_tensor({mir::DataType::FLOAT32, {}}, &one_value);
  auto one = createOp<mir::ops::ConstantOp>(graph, one_tensor)->getOutput(0);
  auto result = createOp<mir::ops::DivOp>(graph, input, one)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertReciprocalV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  convertReciprocal(onnx_node, context);
}

void convertReciprocalV6(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  convertReciprocal(onnx_node, context);
}

} // namespace mir_onnx
