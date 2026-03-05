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

void convertResizeV11(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  const auto coordinate_transformation_mode =
    getAttributeValue<std::string>(onnx_node, "coordinate_transformation_mode", "half_pixel");
  // INFO: As of now only 'nearest' mode is supported, so this attribute is not used.
  // const auto cubic_coeff_a = getAttributeValue<float>(onnx_node, "cubic_coeff_a", -0.75);
  const auto exclude_outside = getAttributeValue<std::int64_t>(onnx_node, "exclude_outside", 0);
  // INFO: As of now only 'nearest' mode is supported, so this attribute is not used.
  // const auto extrapolation_value = getAttributeValue<float>(onnx_node, "extrapolation_value",
  // 0.0);
  const auto mode = getAttributeValue<std::string>(onnx_node, "mode", "nearest");
  const auto nearest_mode =
    getAttributeValue<std::string>(onnx_node, "nearest_mode", "round_prefer_floor");

  if (coordinate_transformation_mode != "half_pixel")
    throw std::runtime_error{
      "Resize v11: Only 'half_pixel' coordinate transformation is supported"};
  if (exclude_outside != 0)
    throw std::runtime_error{"Resize v11: The exclude outside option is not supported"};
  const auto resize_method = getResizeMethod(mode);
  if (nearest_mode != "round_prefer_floor")
    throw std::runtime_error{"Resize v11: Only 'round_prefer_floor' rounding is supported"};

  // Inputs: [0] X, [1] ROI, [2] scales, [3] sizes (optional)
  if (inputs.size() < 3 || inputs.size() > 4)
    throw std::runtime_error{"Resize v11: Expected between 3 and 4 inputs"};

  int rank = inputs[0]->getShape().rank();
  if (rank != 4)
    throw std::runtime_error("Resize v11: Only 4-D input is supported");

  auto *scales = dynamic_cast<mir::ops::ConstantOp *>(inputs[2]->getNode());
  assert(scales && "Scales could be a constant tensor only");
  auto scales_tensor = mir::Tensor<float>(scales->getValue());
  auto scales_tensor_elements = scales_tensor.getShape().numElements();

  std::vector<float> scales_vector;

  // If scales is empty, use sizes input to calculate scales.
  if (scales_tensor_elements == 0)
  {
    if (inputs.size() != 4)
      throw std::runtime_error{"Resize v11: Sizes input is required when scales is empty"};

    auto *sizes = dynamic_cast<mir::ops::ConstantOp *>(inputs[3]->getNode());
    assert(sizes && "Sizes could be a constant tensor only");
    auto sizes_tensor = mir::Tensor<std::int64_t>(sizes->getValue());
    auto sizes_tensor_elements = sizes_tensor.getShape().numElements();

    if (sizes_tensor_elements != rank)
      throw std::runtime_error{"Resize v11: The number of elements of 'sizes' should be the same "
                               "as the rank of input 'X'"};

    auto &input_shape = inputs[0]->getShape();
    scales_vector.resize(sizes_tensor_elements);

    // Calculate scales from sizes
    for (int i = 0; i < sizes_tensor_elements; i++)
    {
      if (input_shape.dim(i) == 0)
        throw std::runtime_error{"Resize v11: Input dimension cannot be zero"};
      scales_vector[i] = static_cast<float>(sizes_tensor.atOffset(i)) / input_shape.dim(i);
      // Verify that the float arithmetic is reversible.
      if (input_shape.dim(i) * scales_vector[i] != sizes_tensor.atOffset(i))
        throw std::runtime_error{"Resize v11: Invalid sizes to scales conversion"};
    }
  }
  else
  {
    if (scales_tensor_elements != rank)
      throw std::runtime_error{"Resize v11: The number of elements of 'scales' should be the same "
                               "as the rank of input 'X'"};
    scales_vector.resize(scales_tensor_elements);
    for (int i = 0; i < scales_tensor_elements; i++)
      scales_vector[i] = scales_tensor.atOffset(i);
  }

  auto result =
    createOp<mir::ops::ResizeOp>(graph, inputs[0], resize_method, scales_vector)->getOutput(0);
  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
