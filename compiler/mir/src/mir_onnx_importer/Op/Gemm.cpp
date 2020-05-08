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

#include "Gemm.h"

#include "ONNXHelpers.h"
#include "AttributeHelpers.h"

#include "mir/TensorUtil.h"

#include "mir/ops/AddOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/FullyConnectedOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/ReshapeOp.h"
#include "mir/ops/TransposeOp.h"

namespace mir_onnx
{

static void convertGemm(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  std::vector<mir::Operation::Output *> inputs = context->getNodeInputs(onnx_node);
  mir::Graph *graph = context->getGraph();

  assert(inputs.size() == 2 || inputs.size() == 3);

  auto a = inputs[0];
  auto b = inputs[1];
  auto c = inputs.size() > 2 ? inputs[2] : nullptr;

  // 1.0f is the default factor.
  const auto alpha_val = getAttributeValue<float>(onnx_node, "alpha", 1.0f);
  const auto beta_val = getAttributeValue<float>(onnx_node, "beta", 1.0f);

  // 0 means that no transpose is needed. It is the default value.
  const auto trans_a = getAttributeValue<std::int64_t>(onnx_node, "transA", 0);
  const auto trans_b = getAttributeValue<std::int64_t>(onnx_node, "transB", 0);

  // Transpose the A and B matrices as needed.
  if (trans_a)
    a = createOp<mir::ops::TransposeOp>(graph, a, std::vector<std::size_t>{1, 0})->getOutput(0);
  if (trans_b)
    b = createOp<mir::ops::TransposeOp>(graph, b, std::vector<std::size_t>{1, 0})->getOutput(0);

  // Calculate A * B.
  auto ab = createOp<mir::ops::FullyConnectedOp>(graph, a, b)->getOutput(0);

  // Multiply A * B by the constant factor.
  if (alpha_val != 1.0f)
  {
    mir::TensorVariant alpha_tensor({mir::DataType::FLOAT32, {}}, &alpha_val);
    auto alpha = createOp<mir::ops::ConstantOp>(graph, alpha_tensor)->getOutput(0);
    ab = createOp<mir::ops::MulOp>(graph, alpha, ab)->getOutput(0);
  }

  // If there are no third input, node is simple A*B multiplication
  if (!c)
  {
    context->setNodeOutputs(onnx_node, {ab});
    return;
  }

  // Multiply C by the constant factor.
  if (beta_val != 1.0f)
  {
    mir::TensorVariant beta_tensor({mir::DataType::FLOAT32, {}}, &beta_val);
    auto beta = createOp<mir::ops::ConstantOp>(graph, beta_tensor)->getOutput(0);
    c = createOp<mir::ops::MulOp>(graph, beta, c)->getOutput(0);
  }

  // Calculate the result: alpha * A * B + beta * C.
  auto result = createOp<mir::ops::AddOp>(graph, ab, c)->getOutput(0);

  context->setNodeOutputs(onnx_node, {result});
}

void convertGemmV1(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  return convertGemm(onnx_node, context);
}

void convertGemmV6(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // This version differs from V1: in description of C input (redundant text "can be inplace.")
  return convertGemm(onnx_node, context);
}

void convertGemmV7(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // This version differs from V6: removed "broadcast" atribute
  return convertGemm(onnx_node, context);
}

void convertGemmV9(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // This version differs from V7: added more supported types
  return convertGemm(onnx_node, context);
}

void convertGemmV11(const onnx::NodeProto &onnx_node, ConverterContext *context)
{
  // This operation differs from V11: input C is optional
  return convertGemm(onnx_node, context);
}

} // namespace mir_onnx
