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

#include "Constant.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/TensorVariant.h"
#include "mir/ops/ConstantOp.h"

namespace mir_onnx
{

void convertConstantV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  const auto onnx_tensor = getAttributeValue<onnx::TensorProto>(onnx_node, "value");
  auto mir_tensor = createTensor(&onnx_tensor);

  auto result = graph->create<mir::ops::ConstantOp>(mir_tensor)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertConstantV9(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // Since version 9 Constant operation support other types contained in tensor
  convertConstantV1(onnx_node, context);
}

void convertConstantV11(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  const auto *value_attr = findAttribute(onnx_node, "value");
  const auto *sparse_value_attr = findAttribute(onnx_node, "sparse_value");
  if (value_attr == nullptr && sparse_value_attr == nullptr)
    throw std::runtime_error("Not enough attributes in Constant operation!");

  if (value_attr != nullptr)
    return convertConstantV9(onnx_node, context);

  if (sparse_value_attr != nullptr)
    throw std::runtime_error("Not supported sparse_tensor in Constant operation!");
}

} // namespace mir_onnx
