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

#include "BatchNormalization.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/ReshapeOp.h"

#include <cmath>

namespace mir_onnx
{

void convertBatchNormalizationV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // consumed_inputs attribute not used
  convertBatchNormalizationV6(onnx_node, context);
}

void convertBatchNormalizationV6(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  const auto is_test = getAttributeValue<std::int64_t>(onnx_node, "is_test", 0);
  if (is_test == 0)
    throw std::runtime_error("Not supported is_test attribute!");

  convertBatchNormalizationV7(onnx_node, context);
}

void convertBatchNormalizationV7(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // spatial attribute used only for learning

  convertBatchNormalizationV9(onnx_node, context);
}

void convertBatchNormalizationV9(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // momentum attrribute used only for learning

  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 5);
  auto input = inputs[0];
  auto scale = inputs[1];
  auto bias = inputs[2];
  auto mean = inputs[3];
  auto var = inputs[4];

  // 1e-05f is the default epsilon.
  const auto epsilon = getAttributeValue<float>(onnx_node, "epsilon", 1e-05f);

  // Y = (X - mean) * scale / sqrt(var + epsilon) + bias =
  //   = (X + C1) * C2 + bias
  // We need these to be constants since we are going to change them.
  // TODO Implement the formula using ops and let the optimizer constant-fold them.
  auto scale_op = dynamic_cast<mir::ops::ConstantOp *>(scale->getNode());
  auto mean_op = dynamic_cast<mir::ops::ConstantOp *>(mean->getNode());
  auto var_op = dynamic_cast<mir::ops::ConstantOp *>(var->getNode());

  if (scale_op == nullptr || mean_op == nullptr || var_op == nullptr)
    throw std::runtime_error(
      "BatchNormalization: only constant 'scale', 'mean' and 'variance' inputs are supported.");

  mir::Tensor<float> scale_accessor(scale_op->getValue());
  mir::Tensor<float> mean_accessor(mean_op->getValue());
  mir::Tensor<float> var_accessor(var_op->getValue());

  // C1 = -mean
  for (const auto &idx : mir::ShapeRange(mean_accessor.getShape()))
    mean_accessor.at(idx) *= -1;

  // C2 = scale / sqrt(var + epsilon)
  for (const auto &idx : mir::ShapeRange(scale_accessor.getShape()))
    scale_accessor.at(idx) /= std::sqrt(var_accessor.at(idx) + epsilon);

  assert(mean_accessor.getShape().rank() == 1);
  auto input_rank = input->getShape().rank();
  if (input_rank < 2)
    throw std::runtime_error("Inputs with shape rank < 2 are not supported for batchnorm");

  mir::Shape new_shape(std::vector<std::int32_t>(input_rank, 1));

  new_shape.dim(1) = mean_accessor.getShape().dim(0); // set channel dim

  auto reshaped_mean = createOp<mir::ops::ReshapeOp>(graph, mean, new_shape)->getOutput(0);
  auto reshaped_scale = createOp<mir::ops::ReshapeOp>(graph, scale, new_shape)->getOutput(0);
  auto reshaped_bias = createOp<mir::ops::ReshapeOp>(graph, bias, new_shape)->getOutput(0);

  // Y = (X + C1) * C2 + bias
  auto result = createOp<mir::ops::AddOp>(graph, input, reshaped_mean)->getOutput(0);
  result = createOp<mir::ops::MulOp>(graph, result, reshaped_scale)->getOutput(0);
  result = createOp<mir::ops::AddOp>(graph, result, reshaped_bias)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

} // namespace mir_onnx
