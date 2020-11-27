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

#include "Upsample.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/Tensor.h"

#include "mir/ops/ConstantOp.h"
#include "mir/ops/ResizeOp.h"

#include <stdexcept>

namespace mir_onnx
{

void convertUpsampleV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  // "nearest" is the default mode.
  std::string mode = getAttributeValue<std::string>(onnx_node, "mode", "nearest");
  assert(mode == "nearest" && "Unsupported upscale mode!");

  const float h_scale = getAttributeValue<float>(onnx_node, "height_scale", 0.0f); // required
  const float w_scale = getAttributeValue<float>(onnx_node, "width_scale", 0.0f);  // required
  if (h_scale < 1.0f || w_scale < 1.0f)
    throw std::runtime_error("Wrong scale attributes!");

  assert(inputs[0]->getShape().rank() == 4 && "Only rank 4 is supported");
  std::vector<float> scales_vector(4);
  // NCHW
  scales_vector.at(0) = 1.0f;
  scales_vector.at(1) = 1.0f;
  scales_vector.at(2) = h_scale;
  scales_vector.at(3) = w_scale;

  auto result =
    createOp<mir::ops::ResizeOp>(graph, inputs[0],
                                 mir::ops::ResizeOp::ResizeMethod::nearestNeighbor, scales_vector)
      ->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertUpsampleV7(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  // "nearest" is the default mode.
  std::string mode = getAttributeValue<std::string>(onnx_node, "mode", "nearest");
  assert(mode == "nearest" && "Unsupported upscale mode!");

  const auto *scales_attr = findAttribute(onnx_node, "scales");
  if (!scales_attr)
    throw std::runtime_error("Not enough required scales attribute!");

  if (scales_attr->floats_size() != inputs[0]->getShape().rank())
    throw std::runtime_error(
      "Number of elements of scales should be the same as the rank of input");

  assert(inputs[0]->getShape().rank() == 4 && "Only rank 4 is supported");
  std::vector<float> scales_vector(4);
  // NCHW
  scales_vector.at(0) = scales_attr->floats(0);
  scales_vector.at(1) = scales_attr->floats(1);
  scales_vector.at(2) = scales_attr->floats(2);
  scales_vector.at(3) = scales_attr->floats(3);

  auto result =
    createOp<mir::ops::ResizeOp>(graph, inputs[0],
                                 mir::ops::ResizeOp::ResizeMethod::nearestNeighbor, scales_vector)
      ->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertUpsampleV9(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  // "nearest" is the default mode.
  const auto mode = getAttributeValue<std::string>(onnx_node, "mode", "nearest");
  if (mode != "nearest")
    throw std::runtime_error("Upsample: only 'nearest' mode is supported.");

  // relies on attributes being lifted to constants (ONNX optimization pass)
  assert(inputs.size() > 1);
  auto *scales = dynamic_cast<mir::ops::ConstantOp *>(inputs[1]->getNode());
  assert(scales && "Weights could be a constant tensor only");
  auto scales_tensor = mir::Tensor<float>(scales->getValue());
  int rank = inputs[0]->getShape().rank();
  if (rank != 4)
    throw std::runtime_error("Upsample: only 4-D input is supported.");
  assert(scales_tensor.getShape().numElements() == rank &&
         "The number of elements of 'scales' should be the same as the rank of input 'X'");
  std::vector<float> scales_vector(rank);
  for (int i = 0; i < scales_tensor.getShape().numElements(); i++)
    scales_vector[i] = scales_tensor.atOffset(i);

  auto result =
    createOp<mir::ops::ResizeOp>(graph, inputs[0],
                                 mir::ops::ResizeOp::ResizeMethod::nearestNeighbor, scales_vector)
      ->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
