/*
 * Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Resize.h"

#include <stdexcept>

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/Tensor.h"

#include "mir/ops/ConstantOp.h"
#include "mir/ops/ResizeOp.h"

namespace mir_onnx
{

static mir::ops::ResizeOp::ResizeMethod getResizeMethod(const std::string &mode)
{
  if (mode == "nearest")
    return mir::ops::ResizeOp::ResizeMethod::nearestNeighbor;
  throw std::runtime_error{"Unsupported mode for Resize operator: " + mode};
}

void convertResizeV10(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  const auto mode = getAttributeValue<std::string>(onnx_node, "mode", "nearest");
  const auto resize_method = getResizeMethod(mode);

  // Inputs: [0] X, [1] scales
  if (inputs.size() != 2)
    throw std::runtime_error{"Resize v10: Expected 2 inputs"};

  int rank = inputs[0]->getShape().rank();
  if (rank != 4)
    throw std::runtime_error("Resize v10: Only 4-D input is supported");

  auto *scales = dynamic_cast<mir::ops::ConstantOp *>(inputs[1]->getNode());
  assert(scales && "Scales could be a constant tensor only");
  auto scales_tensor = mir::Tensor<float>(scales->getValue());
  auto scales_tensor_elements = scales_tensor.getShape().numElements();

  if (scales_tensor_elements != rank)
    throw std::runtime_error{
      "Resize v10: The number of elements of 'scales' should be the same as the rank of input 'X'"};

  std::vector<float> scales_vector(scales_tensor_elements);
  for (int i = 0; i < scales_tensor_elements; i++)
    scales_vector[i] = scales_tensor.atOffset(i);

  auto result =
    createOp<mir::ops::ResizeOp>(graph, inputs[0], resize_method, scales_vector)->getOutput(0);
  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
